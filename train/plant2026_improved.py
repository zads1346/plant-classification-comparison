"""2026task1: leakage-aware image classification, no test-label/pseudo-label use.
Run --self-test first. On Kaggle, confirm the pretrained policy, select GPU,
then run --preset fast/full. This file does not install or download datasets.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

CODE_VERSION = "2026task1-clean-v1.1"
CLASS_NAMES = ["Black-grass", "Common wheat", "Loose Silky-bent", "Scentless Mayweed", "Sugar beet"]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
FILL = tuple(int(round(float(v) * 255)) for v in MEAN)


@dataclass
class Config:
    data_root: str = "/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2"
    template_path: str = "/kaggle/input/competitions/2026task1/submission-for-task2.csv"
    output_dir: str = "/kaggle/working/plant2026_improved_fast"
    preset: str = "fast"
    architectures: tuple = ("efficientnet_b2",)
    folds_to_run: tuple = (0,)
    n_splits: int = 5
    image_size: int = 288
    epochs: int = 24
    min_epochs: int = 10
    patience: int = 7
    head_epochs: int = 2
    batch_size: int = 16
    workers: int = 2
    backbone_lr: float = 1.0e-4
    head_lr: float = 6.0e-4
    weight_decay: float = 1.0e-4
    label_smoothing: float = 0.03
    seed: int = 42
    # The public rules do not explicitly settle external pretrained weights.
    # Set True only after confirming with the instructor/organizer.
    pretrained_allowed: bool = False
    use_pretrained: bool = True
    weights_dir: str = ""  # optional folder with official torchvision .pth files
    require_gpu: bool = True
    freeze_batchnorm: bool = True
    use_ema: bool = True
    ema_decay: float = 0.99
    # Identical policy is used for EVERY validation epoch, saved OOF, and test.
    # 'none' is the conservative default; 'hflip' and 'd4' are explicit ablations.
    tta: str = "none"
    resize_mode: str = "stretch"  # retains the original single-view geometry; letterbox is an ablation
    resume_completed_folds: bool = True
    groups_csv: str = ""  # optional columns: train_relpath, group_id


def make_config(preset="fast", **overrides):
    cfg = Config(preset=preset)
    if preset == "full":
        cfg = replace(cfg, architectures=("efficientnet_b2", "convnext_tiny"),
                      folds_to_run=tuple(range(5)), epochs=28,
                      output_dir="/kaggle/working/plant2026_improved_full")
    elif preset != "fast":
        raise ValueError("preset must be 'fast' or 'full'")
    return replace(cfg, **overrides)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(cfg):
    require(cfg.n_splits >= 2, "n_splits must be at least 2")
    require(len(cfg.folds_to_run) > 0 and len(set(cfg.folds_to_run)) == len(cfg.folds_to_run),
            "folds_to_run must be nonempty and unique")
    require(all(0 <= f < cfg.n_splits for f in cfg.folds_to_run), "Invalid fold index")
    require(len(cfg.architectures) > 0 and len(set(cfg.architectures)) == len(cfg.architectures),
            "architectures must be nonempty and unique")
    require(all(a in {"efficientnet_b2", "convnext_tiny"} for a in cfg.architectures),
            "Unsupported architecture")
    require(cfg.epochs > cfg.head_epochs >= 0, "epochs must exceed head_epochs")
    require(1 <= cfg.min_epochs <= cfg.epochs, "Invalid min_epochs")
    require(cfg.batch_size >= 2 and cfg.workers >= 0 and cfg.image_size >= 32, "Invalid data config")
    require(cfg.tta in {"none", "hflip", "d4"}, "Invalid TTA")
    require(cfg.resize_mode in {"letterbox", "stretch"}, "Invalid resize mode")
    require(0 <= cfg.label_smoothing < 1 and 0 <= cfg.ema_decay < 1, "Invalid regularization")
    require(cfg.patience > 0 and cfg.backbone_lr > 0 and cfg.head_lr > 0, "Invalid optimizer config")
    require(not cfg.use_pretrained or cfg.pretrained_allowed,
            "Pretrained policy is unconfirmed. Confirm ImageNet weights are permitted, then set "
            "pretrained_allowed=True / --allow-pretrained. If prohibited, use use_pretrained=False "
            "/ --scratch (performance is not equivalent; retune epochs/LR). No weights were downloaded.")


def read_rgb(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, FILL + (255,))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        return image.copy()


def image_hash(path):
    image = read_rgb(path)
    digest = hashlib.sha256()
    digest.update(str(image.size).encode())
    digest.update(image.tobytes())
    return digest.hexdigest(), image.width, image.height


def audit_data(cfg):
    """Read image bytes, class names, template IDs; never obtain test labels."""
    root = Path(cfg.data_root)
    train_dir, test_dir = root / "train", root / "test"
    require(train_dir.is_dir() and test_dir.is_dir(), f"Missing train/test under {root}")
    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    require(classes == CLASS_NAMES, f"Unexpected folder classes: {classes}; inspect before changing mapping")
    rows = []
    for label, name in enumerate(classes):
        paths = sorted(p for p in (train_dir / name).rglob("*") if p.suffix.lower() in EXTENSIONS and p.is_file())
        require(len(paths) >= cfg.n_splits, f"Too few samples in {name}")
        for p in paths:
            digest, width, height = image_hash(p)
            rows.append(dict(path=str(p), train_relpath=p.relative_to(train_dir).as_posix(),
                             label=label, category=name, image_hash=digest, width=width, height=height))
    train = pd.DataFrame(rows)
    conflicts = train.groupby("image_hash")["label"].nunique()
    require(not (conflicts > 1).any(), "Identical decoded training images have conflicting labels. Review source data.")
    template = pd.read_csv(cfg.template_path, dtype=str, keep_default_na=False)
    require(template.columns.tolist() == ["ID", "Category"], "Template must have exactly ID,Category columns")
    require(len(template) > 0 and template.ID.is_unique and template.ID.str.len().min() > 0,
            "Template IDs must be nonempty and unique")
    test_rows = []
    for image_id in template.ID:
        rel = Path(image_id)
        require(not rel.is_absolute() and ".." not in rel.parts, f"Unsafe ID path: {image_id}")
        p = test_dir / rel
        require(p.is_file() and p.suffix.lower() in EXTENSIONS, f"Missing test image: {p}")
        digest, width, height = image_hash(p)
        test_rows.append(dict(path=str(p), ID=image_id, image_hash=digest, width=width, height=height))
    test = pd.DataFrame(test_rows)
    actual_ids = {p.relative_to(test_dir).as_posix() for p in test_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in EXTENSIONS}
    require(set(template.ID) == actual_ids, "Test image IDs and template IDs differ; inspect inputs")
    groups = make_groups(train, cfg.groups_csv)
    train["group"] = groups
    split_method = StratifiedGroupKFold if train.group.nunique() < len(train) else StratifiedKFold
    splitter = split_method(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    iterator = (splitter.split(np.zeros(len(train)), train.label, train.group)
                if split_method is StratifiedGroupKFold else splitter.split(np.zeros(len(train)), train.label))
    fold_id = np.full(len(train), -1, dtype=np.int64)
    splits = []
    for fold, (tr, va) in enumerate(iterator):
        require(set(train.iloc[tr].group).isdisjoint(set(train.iloc[va].group)), "Group leakage")
        require(set(train.iloc[va].label) == set(range(len(classes))),
                "A validation fold lacks a class. Use fewer folds or review groups.")
        require(set(train.iloc[tr].label) == set(range(len(classes))), "Training fold lacks a class")
        fold_id[va] = fold
        splits.append((tr, va))
    require((fold_id >= 0).all(), "Incomplete fold assignment")
    train["fold"] = fold_id
    train_hashes = set(train.image_hash)
    overlap = test[test.image_hash.isin(train_hashes)][["ID", "image_hash"]]
    summary = {"train_count": len(train), "test_count": len(test),
               "class_counts": train.category.value_counts().sort_index().to_dict(),
               "duplicate_training_images": int(train.image_hash.duplicated().sum()),
               "train_test_exact_overlap": int(len(overlap)), "splitter": split_method.__name__}
    print("Data audit:", json.dumps(summary, ensure_ascii=False))
    if len(overlap):
        print("WARNING: train/test exact-image overlap detected. Recorded for review; labels are NOT copied to test.")
    return train, test, template, splits, summary, overlap


def make_groups(train, groups_csv=""):
    # Union exact image duplicates with optional specimen/capture-session groups.
    parents = list(range(len(train)))
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    def union_keys(keys):
        first = {}
        for i, key in enumerate(keys):
            if key in first:
                parents[find(i)] = find(first[key])
            else:
                first[key] = i
    union_keys(train.image_hash)
    if groups_csv:
        g = pd.read_csv(groups_csv, dtype=str, keep_default_na=False)
        require(set(g.columns) == {"train_relpath", "group_id"}, "groups_csv columns: train_relpath,group_id")
        require(g.train_relpath.is_unique and g.group_id.str.len().min() > 0, "Invalid specimen groups")
        require(set(g.train_relpath) == set(train.train_relpath), "groups_csv must cover all training images exactly")
        mapping = dict(zip(g.train_relpath, g.group_id))
        union_keys([mapping[p] for p in train.train_relpath])
    return [find(i) for i in range(len(train))]


def image_array(image, size, training=False, resize_mode="letterbox"):
    """Whole-image processing. No data-dependent crop or normalization fitting."""
    if training:
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        if random.random() < 0.5:
            image = ImageOps.flip(image)
        image = image.rotate(random.choice([0, 90, 180, 270]), expand=True)
        # expand=True retains leaves instead of clipping image corners.
        if random.random() < 0.5:
            image = image.rotate(random.uniform(-15, 15), resample=Image.Resampling.BILINEAR,
                                 expand=True, fillcolor=FILL)
        for enhancer, strength in [(ImageEnhance.Brightness, 0.12),
                                   (ImageEnhance.Contrast, 0.12), (ImageEnhance.Color, 0.10)]:
            image = enhancer(image).enhance(random.uniform(1-strength, 1+strength))
    if resize_mode == "letterbox":
        scale = size / max(image.width, image.height)
        width = min(size, max(1, round(image.width * scale)))
        height = min(size, max(1, round(image.height * scale)))
        fitted = image.resize((width, height), Image.Resampling.BICUBIC)
        image = Image.new("RGB", (size, size), FILL)
        image.paste(fitted, ((size-width)//2, (size-height)//2))
    else:
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return np.ascontiguousarray(((pixels - MEAN) / STD).transpose(2, 0, 1))


class PlantDataset:
    def __init__(self, frame, cfg, training=False):
        self.frame = frame.reset_index(drop=True)
        self.cfg, self.training = cfg, training
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        array = image_array(read_rgb(row.path), self.cfg.image_size, self.training, self.cfg.resize_mode)
        return array, int(row.label) if "label" in self.frame else -1


def metrics(y, probabilities):
    validate_probabilities(probabilities, len(y), len(CLASS_NAMES))
    prediction = probabilities.argmax(1)
    return {"micro_f1": float(f1_score(y, prediction, average="micro")),
            "accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=range(len(CLASS_NAMES)), average="macro", zero_division=0)),
            "nll": float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1)).mean())}


def validate_probabilities(probabilities, n, k):
    p = np.asarray(probabilities)
    require(p.shape == (n, k), f"Probability shape {p.shape} != {(n, k)}")
    require(np.isfinite(p).all() and (p >= 0).all(), "Invalid probabilities")
    require(np.allclose(p.sum(1), 1.0, atol=1e-4), "Probability rows must sum to one")


def save_submission(template, probabilities, path):
    validate_probabilities(probabilities, len(template), len(CLASS_NAMES))
    output = template.copy()
    output["Category"] = np.asarray(CLASS_NAMES)[probabilities.argmax(1)]
    require(output.ID.tolist() == template.ID.tolist(), "Submission order changed")
    require(set(output.Category).issubset(CLASS_NAMES) and output.Category.notna().all(), "Invalid category mapping")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    output.to_csv(temporary, index=False)
    os.replace(temporary, path)
    reread = pd.read_csv(path, dtype=str, keep_default_na=False)
    require(reread.equals(output.astype(str)), "Submission CSV read-back mismatch")
    return output


def load_torch():
    global torch, nn
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for training. Use Kaggle's standard GPU image; do not blindly upgrade torch/torchvision.") from exc


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    import torch
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_loader(frame, cfg, training, seed, device):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(PlantDataset(frame, cfg, training), batch_size=cfg.batch_size,
                                      shuffle=training, num_workers=cfg.workers, pin_memory=device.type == "cuda",
                                      drop_last=False, worker_init_fn=seed_worker, generator=generator,
                                      persistent_workers=False)


def build_model(architecture, cfg, num_classes=5):
    from torchvision import models
    builders = {"efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1,
                                    "efficientnet_b2_rwightman-c35c1473.pth"),
                "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                                  "convnext_tiny-983f1562.pth")}
    builder, weights, filename = builders[architecture]
    if cfg.use_pretrained and cfg.weights_dir:
        path = Path(cfg.weights_dir) / filename
        require(path.is_file(), f"Missing official weights: {path}")
        model = builder(weights=None)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    else:
        try:
            model = builder(weights=weights if cfg.use_pretrained else None)
        except Exception as exc:
            raise RuntimeError("Model/weight loading failed. Check torch/torchvision compatibility and, only if rules permit, "
                               "Internet or weights_dir. Never silently fall back to random weights.") from exc
    features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(features, num_classes)
    return model


def set_trainability(model, head_only):
    head_ids = {id(p) for p in model.classifier[-1].parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(not head_only or id(parameter) in head_ids)


def make_optimizer(model, cfg):
    head_ids = {id(p) for p in model.classifier[-1].parameters()}
    return torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": cfg.backbone_lr,
         "base_lr": cfg.backbone_lr},
        {"params": list(model.classifier[-1].parameters()), "lr": cfg.head_lr, "base_lr": cfg.head_lr},
    ], weight_decay=cfg.weight_decay)


def set_learning_rates(optimizer, epoch, cfg):
    # Head phase first; two-epoch backbone warmup followed by cosine decay.
    head_phase = cfg.head_epochs if cfg.use_pretrained else 0
    if epoch < head_phase:
        factor = 1.0
    else:
        t, total = epoch - head_phase, max(1, cfg.epochs - head_phase)
        warmup = min(2, total)
        if t < warmup:
            factor = (t + 1) / warmup
        else:
            progress = (t - warmup) / max(1, total - warmup - 1)
            factor = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, progress)))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor


def tta_views(images, policy):
    if policy == "none":
        return [images]
    if policy == "hflip":
        return [images, torch.flip(images, dims=[3])]
    # All eight UNIQUE D4 transforms; no duplicated 180-degree rotation.
    return [torch.rot90(images, k, dims=[2, 3]) for k in range(4)] + [
        torch.rot90(torch.flip(images, dims=[3]), k, dims=[2, 3]) for k in range(4)]


def predict(model, loader, cfg, device):
    model.eval()
    all_probabilities = []
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            probabilities = []
            for view in tta_views(images, cfg.tta):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(view)
                probabilities.append(logits.float().softmax(1))
            all_probabilities.append(torch.stack(probabilities).mean(0).cpu().numpy())
    result = np.concatenate(all_probabilities)
    validate_probabilities(result, len(loader.dataset), len(CLASS_NAMES))
    return result


def update_ema(ema, model, step, decay):
    decay = min(decay, (1 + step) / (10 + step))
    with torch.no_grad():
        state = model.state_dict()
        for name, value in ema.state_dict().items():
            if value.is_floating_point():
                value.mul_(decay).add_(state[name], alpha=1-decay)
            else:
                value.copy_(state[name])


def train_epoch(model, loader, optimizer, scaler, criterion, cfg, device, head_only, ema=None, step=0):
    model.train()
    if head_only or (cfg.use_pretrained and cfg.freeze_batchnorm):
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # A lowered AMP scale means the optimizer step was skipped.
        if scaler.get_scale() >= previous_scale:
            step += 1
            if ema is not None:
                update_ema(ema, model, step, cfg.ema_decay)
        loss_sum += loss.item() * len(labels)
        correct += int((logits.argmax(1) == labels).sum())
        total += len(labels)
    return {"train_loss": loss_sum/total, "train_augmented_accuracy": correct/total}, step


def json_write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def fingerprint(cfg, train, test):
    configuration = asdict(cfg)
    for field in ["output_dir", "resume_completed_folds"]:
        configuration.pop(field)
    import torchvision, sklearn, PIL
    local_weight_hashes = {}
    if cfg.use_pretrained and cfg.weights_dir:
        for name in cfg.architectures:
            filename = {"efficientnet_b2": "efficientnet_b2_rwightman-c35c1473.pth",
                        "convnext_tiny": "convnext_tiny-983f1562.pth"}[name]
            path = Path(cfg.weights_dir) / filename
            require(path.is_file(), f"Missing official weights: {path}")
            local_weight_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"code_version": CODE_VERSION, "config": configuration,
               "local_weight_hashes": local_weight_hashes,
               "versions": {"torchvision": torchvision.__version__, "sklearn": sklearn.__version__,
                            "PIL": PIL.__version__, "numpy": np.__version__, "pandas": pd.__version__},
               "train": train[["train_relpath", "image_hash", "label", "fold", "group"]].to_dict("records"),
               "test": test[["ID", "image_hash"]].to_dict("records"),
               "torch": torch.__version__}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def fit_fold(architecture, cfg, train, test, train_idx, valid_idx, fold, out, run_id, device):
    folder = out / architecture / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    done_path, prob_path, checkpoint = folder/"complete.json", folder/"probabilities.npz", folder/"best.pt"
    if cfg.resume_completed_folds and done_path.is_file() and prob_path.is_file() and checkpoint.is_file():
        done = json.loads(done_path.read_text())
        require(done.get("fingerprint") == run_id, "Resume fingerprint mismatch; choose a new output_dir")
        with np.load(prob_path, allow_pickle=False) as saved:
            require(np.array_equal(saved["valid_indices"], valid_idx), "Resume validation indices mismatch")
            vp, tp = saved["valid"].copy(), saved["test"].copy()
        validate_probabilities(vp, len(valid_idx), 5)
        validate_probabilities(tp, len(test), 5)
        print(f"Reusing completed {architecture}, fold {fold}")
        return vp, tp, done
    seed_everything(cfg.seed + fold)
    tr_loader = make_loader(train.iloc[train_idx], cfg, True, cfg.seed+fold, device)
    va_loader = make_loader(train.iloc[valid_idx], cfg, False, cfg.seed+fold, device)
    te_loader = make_loader(test, cfg, False, cfg.seed+fold, device)
    model = build_model(architecture, cfg).to(device)
    ema = copy.deepcopy(model).eval() if cfg.use_ema else None
    if ema is not None:
        for p in ema.parameters():
            p.requires_grad_(False)
    optimizer = make_optimizer(model, cfg)
    scaler = make_scaler(enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    best_score, best_nll, stale, step, history, best_epoch = -1., float("inf"), 0, 0, [], -1
    y_valid = train.iloc[valid_idx].label.to_numpy()
    for epoch in range(cfg.epochs):
        head_only = cfg.use_pretrained and epoch < cfg.head_epochs
        set_trainability(model, head_only)
        set_learning_rates(optimizer, epoch, cfg)
        train_stats, step = train_epoch(model, tr_loader, optimizer, scaler, criterion, cfg,
                                       device, head_only, ema, step)
        eval_model = ema if ema is not None else model
        p_valid = predict(eval_model, va_loader, cfg, device)
        valid = metrics(y_valid, p_valid)
        row = {"epoch": epoch+1, **train_stats, **valid, "backbone_lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"{architecture} fold={fold+1}/{cfg.n_splits} epoch={epoch+1:02d} "
              f"train_loss={row['train_loss']:.4f} microF1={valid['micro_f1']:.4f} "
              f"macroF1={valid['macro_f1']:.4f} nll={valid['nll']:.4f}")
        better = valid["micro_f1"] > best_score + 1e-12 or (
            abs(valid["micro_f1"] - best_score) <= 1e-12 and valid["nll"] < best_nll - 1e-6)
        if better:
            best_score, best_nll, stale, best_epoch = valid["micro_f1"], valid["nll"], 0, epoch+1
            state = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
            tmp = checkpoint.with_suffix(".tmp")
            torch.save({"model": state, "fingerprint": run_id, "architecture": architecture,
                        "classes": CLASS_NAMES, "epoch": best_epoch, "metrics": valid}, tmp)
            os.replace(tmp, checkpoint)
        else:
            stale += 1
        pd.DataFrame(history).to_csv(folder/"history.csv", index=False)
        if epoch+1 >= cfg.min_epochs and stale >= cfg.patience:
            print("Early stopping")
            break
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(saved["fingerprint"] == run_id and saved["classes"] == CLASS_NAMES, "Checkpoint identity mismatch")
    model.load_state_dict(saved["model"], strict=True)
    p_valid = predict(model, va_loader, cfg, device)
    final_metrics = metrics(y_valid, p_valid)
    require(abs(final_metrics["micro_f1"] - best_score) < 1e-10, "Reloaded checkpoint metric mismatch")
    p_test = predict(model, te_loader, cfg, device)
    with open(prob_path.with_suffix(".tmp"), "wb") as handle:
        np.savez_compressed(handle, valid=p_valid, test=p_test, valid_indices=valid_idx)
    os.replace(prob_path.with_suffix(".tmp"), prob_path)
    done = {"fingerprint": run_id, "architecture": architecture, "fold": fold,
            "best_epoch": best_epoch, "validation_count": len(valid_idx), "metrics": final_metrics}
    json_write(done_path, done)
    del model, ema, optimizer, scaler, tr_loader, va_loader, te_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return p_valid, p_test, done


def run(cfg):
    validate_config(cfg)
    load_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.require_gpu and device.type != "cuda":
        raise RuntimeError("GPU not detected. In Kaggle Settings select a GPU, restart, and verify torch.cuda.is_available().")
    import torchvision
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    print("Versions:", {"torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__})
    started = time.time()
    train, test, template, splits, audit, overlap = audit_data(cfg)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = fingerprint(cfg, train, test)
    meta = out/"run_config.json"
    if meta.exists():
        require(json.loads(meta.read_text()).get("fingerprint") == run_id,
                "This output_dir belongs to different data/config. Choose a NEW output_dir; nothing overwritten.")
    json_write(meta, {"fingerprint": run_id, "config": asdict(cfg), "code_version": CODE_VERSION,
                      "metric": "micro_f1 (=accuracy)", "classes": CLASS_NAMES,
                      "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__}})
    train.to_csv(out/"train_manifest.csv", index=False)
    test.to_csv(out/"test_manifest.csv", index=False)
    overlap.to_csv(out/"train_test_overlap.csv", index=False)
    json_write(out/"data_audit.json", audit)
    oof_by_arch, test_by_arch, fold_results = [], [], []
    covered = train.fold.isin(cfg.folds_to_run).to_numpy()
    for architecture in cfg.architectures:
        oof = np.full((len(train), 5), np.nan, dtype=np.float32)
        test_list = []
        for fold in cfg.folds_to_run:
            tr, va = splits[fold]
            vp, tp, result = fit_fold(architecture, cfg, train, test, tr, va, fold, out, run_id, device)
            oof[va] = vp
            test_list.append(tp)
            fold_results.append(result)
        test_probs = np.mean(test_list, axis=0)
        score = metrics(train.label.to_numpy()[covered], oof[covered])
        print(architecture, "covered validation:", score)
        # Each architecture's predictions remain available for diagnosis.
        save_submission(template, test_probs, out/f"submission_{architecture}.csv")
        oof_by_arch.append(oof)
        test_by_arch.append(test_probs)
    # Predeclared equal weights. No fitting weights/biases on reported OOF labels.
    p_oof = np.full((len(train), 5), np.nan, dtype=np.float32)
    p_oof[covered] = np.mean([p[covered] for p in oof_by_arch], axis=0)
    p_test = np.mean(test_by_arch, axis=0)
    y = train.label.to_numpy()[covered]
    score = metrics(y, p_oof[covered])
    scope = "full OOF" if covered.all() else "partial OOF / one holdout; NOT full cross-validation"
    summary = {"scope": scope, "covered_train_count": int(covered.sum()), "total_train_count": len(train),
               "metrics": score, "folds": fold_results, "ensemble": "equal architecture / equal fold weights",
               "tta": cfg.tta, "pseudo_labels_used": False, "runtime_minutes": (time.time()-started)/60,
               "caution": "Validation folds selected early-stopping epochs; repeated model choices can still overfit CV. "
                          "This is not an untouched final test score. Real Kaggle score is unknown."}
    json_write(out/"metrics.json", summary)
    oof_table = train[["train_relpath", "category", "label", "fold"]].copy()
    oof_table["covered"] = covered
    oof_table["predicted_category"] = ""
    oof_table.loc[covered, "predicted_category"] = np.asarray(CLASS_NAMES)[p_oof[covered].argmax(1)]
    for i, name in enumerate(CLASS_NAMES):
        oof_table[f"p_{name}"] = p_oof[:, i]
    oof_table.to_csv(out/"oof_predictions.csv", index=False)
    oof_table[covered & (oof_table.category != oof_table.predicted_category)].to_csv(out/"oof_errors.csv", index=False)
    pd.DataFrame(confusion_matrix(y, p_oof[covered].argmax(1), labels=range(5)),
                 index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out/"confusion_matrix.csv")
    report = classification_report(y, p_oof[covered].argmax(1), labels=range(5), target_names=CLASS_NAMES,
                                   zero_division=0, output_dict=True)
    json_write(out/"classification_report.json", report)
    np.savez_compressed(out/"ensemble_probabilities.npz", oof=p_oof, test=p_test, covered=covered)
    submission = save_submission(template, p_test, out/"submission.csv")
    print("\nRESULT:", scope, score)
    print("Saved:", out/"submission.csv", "rows:", len(submission))
    print("No real-test score is available here. Do not assume a gain until validated.")
    return summary


def create_synthetic_dataset(folder):
    """Artificial pixels ONLY. Never presented as actual competition performance."""
    folder = Path(folder)
    for ci, name in enumerate(CLASS_NAMES):
        d = folder/"train"/name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            rng = np.random.default_rng(1000*ci+i)
            array = rng.integers(0, 256, (19+i, 31+ci, 3), dtype=np.uint8)
            Image.fromarray(array).save(d/f"image_{i:02}.png")
    (folder/"test").mkdir()
    ids = ["003.png", "001.png", "002.png"]
    for i, name in enumerate(ids):
        Image.new("RGB", (13+i*2, 22), (30, 60+i, 90)).save(folder/"test"/name)
    template = folder/"sample.csv"
    pd.DataFrame({"ID": ids, "Category": ["", "", ""]}).to_csv(template, index=False)
    return template


def self_test():
    """Runs without torch, downloads, real data, or a GPU."""
    with tempfile.TemporaryDirectory(prefix="plant_synthetic_") as tmp:
        template_path = create_synthetic_dataset(tmp)
        cfg = make_config(data_root=tmp, template_path=str(template_path), workers=0)
        train, test, template, splits, audit, _ = audit_data(cfg)
        require(audit["train_count"] == 50 and audit["test_count"] == 3, "Synthetic count failure")
        require(test.ID.tolist() == ["003.png", "001.png", "002.png"], "Template ordering failure")
        image = Image.new("RGB", (20, 50), (255, 0, 0))
        for mode in ["letterbox", "stretch"]:
            for training in [False, True]:
                array = image_array(image, 64, training, mode)
                require(array.shape == (3, 64, 64) and np.isfinite(array).all(), "Transform failure")
        for extreme_size in [(1, 1000), (1000, 1)]:
            require(image_array(Image.new("RGB", extreme_size), 64, False, "letterbox").shape == (3, 64, 64),
                    "Extreme aspect ratio preprocessing failure")
        first = image_array(image, 64)
        require(np.array_equal(first, image_array(image, 64)), "Evaluation is not deterministic")
        groups = pd.DataFrame({"image_hash": ["a", "a", "b", "c"],
                               "train_relpath": ["a", "b", "c", "d"]})
        g = make_groups(groups)
        require(g[0] == g[1] and g[1] != g[2], "Duplicate grouping failure")
        groups_file = Path(tmp)/"groups.csv"
        pd.DataFrame({"train_relpath": ["a", "b", "c", "d"], "group_id": ["x", "y", "y", "z"]}).to_csv(groups_file, index=False)
        g = make_groups(groups, str(groups_file))
        require(g[0] == g[1] == g[2] and g[2] != g[3], "Group union failure")
        p = np.eye(5, dtype=np.float32)[[4, 0, 2]]
        output = save_submission(template, p, Path(tmp)/"out.csv")
        require(output.Category.tolist() == [CLASS_NAMES[4], CLASS_NAMES[0], CLASS_NAMES[2]], "Class mapping failure")
        require(metrics(np.array([4, 0, 2]), p)["accuracy"] == 1, "Metric failure")
        for bad in [np.full((3, 5), np.nan), np.zeros((3, 5)), p[:, :4]]:
            try:
                validate_probabilities(bad, 3, 5)
            except ValueError:
                pass
            else:
                raise AssertionError("Malformed probabilities accepted")
        require(sum(len(va) for _, va in splits) == 50, "Split coverage failure")
        require(all(set(tr).isdisjoint(va) for tr, va in splits), "Index leakage")
        # Actual duplicate pixel files are grouped in the full audit route.
        import shutil
        original = Path(tmp)/"train"/CLASS_NAMES[0]/"image_00.png"
        shutil.copyfile(original, original.with_name("duplicate.png"))
        grouped, _, _, _, duplicate_audit, _ = audit_data(cfg)
        dup = grouped[grouped.image_hash.duplicated(False)]
        require(len(dup) == 2 and dup.fold.nunique() == 1, "Duplicate split leakage")
        require(duplicate_audit["splitter"] == "StratifiedGroupKFold", "Wrong duplicate splitter")
    print("SELF-TEST PASSED: synthetic data audit, class/ID mapping, transforms, grouping, splits, metrics, CSV read-back.")
    print("This does not train a network and is not an accuracy result.")


def torch_smoke_test():
    """Optional Kaggle CPU smoke test: real tensor backward/checkpoint/TTA, synthetic inputs."""
    load_torch()
    device = torch.device("cpu")
    seed_everything(7)
    with tempfile.TemporaryDirectory(prefix="plant_torch_synthetic_") as tmp:
        template_path = create_synthetic_dataset(tmp)
        cfg = make_config(data_root=tmp, template_path=str(template_path), image_size=32,
                          batch_size=8, workers=0, require_gpu=False, use_pretrained=False, tta="d4")
        train, test, template, splits, _, _ = audit_data(cfg)
        model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                              nn.Flatten(), nn.Linear(8, 5))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = make_scaler(enabled=False)
        loader = make_loader(train.iloc[splits[0][0]], cfg, True, 7, device)
        stats, _ = train_epoch(model, loader, optimizer, scaler, nn.CrossEntropyLoss(), cfg, device, False)
        require(math.isfinite(stats["train_loss"]), "Synthetic torch loss failure")
        marker = torch.arange(12).reshape(1, 1, 3, 4)
        require(len({(tuple(v.shape), v.numpy().tobytes()) for v in tta_views(marker, "d4")}) == 8,
                "D4 transforms are not unique")
        path = Path(tmp)/"synthetic.pt"
        torch.save(model.state_dict(), path)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        p = predict(model, make_loader(test, cfg, False, 7, device), cfg, device)
        save_submission(template, p, Path(tmp)/"synthetic_submission.csv")
    print("TORCH SMOKE TEST PASSED. Synthetic tiny network only; no pretrained architecture/GPU/real-data validation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["fast", "full"], default="fast")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--torch-smoke-test", action="store_true")
    parser.add_argument("--allow-pretrained", action="store_true", help="Only after confirming organizer permission")
    parser.add_argument("--scratch", action="store_true", help="No external weights; needs separate LR/epoch tuning")
    parser.add_argument("--data-root")
    parser.add_argument("--template-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--weights-dir")
    parser.add_argument("--groups-csv")
    parser.add_argument("--tta", choices=["none", "hflip", "d4"], default="none")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.torch_smoke_test:
        torch_smoke_test()
        return
    overrides = {k: getattr(args, k) for k in ["data_root", "template_path", "output_dir", "weights_dir", "groups_csv"]
                 if getattr(args, k) is not None}
    cfg = make_config(args.preset, pretrained_allowed=args.allow_pretrained, use_pretrained=not args.scratch,
                      require_gpu=not args.allow_cpu, tta=args.tta, **overrides)
    if args.scratch:
        print("WARNING: training from scratch on 500 labels is not equivalent to the transfer-learning recipe.")
    run(cfg)


if __name__ == "__main__":
    main()
