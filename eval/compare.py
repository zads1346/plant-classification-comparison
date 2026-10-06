"""Offline comparison of labeled predictions. Standard-library only; no training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

CLASSES = ["Black-grass", "Common wheat", "Loose Silky-bent", "Scentless Mayweed", "Sugar beet"]
FIELDS = ["ID", "true_label", "pred_label", "fold", "split_id", "source_kind"]


def read_predictions(path: Path, source_kind: str):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"{path.name}: required columns: {', '.join(FIELDS)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name}: empty predictions")
    ids = set()
    for row in rows:
        if any(row.get(k) is None or not row[k].strip() for k in FIELDS):
            raise ValueError(f"{path.name}: missing/blank field")
        if row["ID"] in ids:
            raise ValueError(f"{path.name}: duplicate ID")
        ids.add(row["ID"])
        if row["true_label"] not in CLASSES or row["pred_label"] not in CLASSES:
            raise ValueError(f"{path.name}: unknown class label")
        if row["source_kind"] != source_kind:
            raise ValueError(f"{path.name}: source_kind does not match config")
    if len({r["split_id"] for r in rows}) != 1:
        raise ValueError(f"{path.name}: mixed split_id values")
    return sorted(rows, key=lambda r: r["ID"])


def alignment_key(rows):
    return [(r["ID"], r["true_label"], r["fold"], r["split_id"]) for r in rows]


def compute_metrics(rows):
    cm = [[0 for _ in CLASSES] for _ in CLASSES]
    index = {c: i for i, c in enumerate(CLASSES)}
    for r in rows:
        cm[index[r["true_label"]]][index[r["pred_label"]]] += 1
    per_class = []
    for i, name in enumerate(CLASSES):
        tp = cm[i][i]
        support = sum(cm[i])
        predicted = sum(row[i] for row in cm)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class.append(dict(name=name, precision=precision, recall=recall, f1=f1, support=support))
    correct = sum(cm[i][i] for i in range(len(CLASSES)))
    return dict(n=len(rows), correct=correct, errors=len(rows)-correct,
                accuracy=correct/len(rows), micro_f1=correct/len(rows),
                macro_f1=sum(r["f1"] for r in per_class)/len(CLASSES),
                confusion_matrix=cm, per_class=per_class)


def build_report(config_path: Path):
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    kind = config.get("source_kind")
    if kind not in {"synthetic", "validation"}:
        raise ValueError("source_kind must be synthetic or validation")
    models = config.get("models", [])
    if not 2 <= len(models) <= 8:
        raise ValueError("Compare 2–8 models")
    if len({m["name"] for m in models}) != len(models):
        raise ValueError("Model names must be unique")
    report = dict(title=config.get("title", "植物分类实验对比"), source_kind=kind,
                  scope=config.get("scope", "用户提供的同一验证集"), classes=CLASSES, models=[],
                  caveat="这些数字来自合成预测，仅展示工具功能，不代表训练、模型或比赛成绩。" if kind == "synthetic"
                  else "这些数字来自用户提供的有标签验证预测；不是未知标签测试集成绩，数据来源与训练隔离须另行核验。")
    first = None
    for model in models:
        path = (config_path.parent / model["path"]).resolve()
        rows = read_predictions(path, kind)
        key = alignment_key(rows)
        if first is None:
            first = key
        elif key != first:
            raise ValueError("Models must have exactly the same IDs, true labels, fold assignments and split_id. Do not compare one-fold and full-OOF scores.")
        report["models"].append(dict(name=model["name"], note=model.get("note", ""),
                                     sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                     metrics=compute_metrics(rows), rows=rows))
    report["split_id"] = report["models"][0]["rows"][0]["split_id"]
    return report


def write_report(report, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    template = (Path(__file__).parent.parent / "preview" / "template.html").read_text(encoding="utf-8")
    # Escaping '<' prevents data containing </script> from closing the JSON element.
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    rendered = template.replace("__REPORT_JSON__", payload)
    (output / "index.html").write_text(rendered, encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "models"}
    summary["models"] = [{k: v for k, v in m.items() if k != "rows"} for m in report["models"]]
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (output / "errors.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model"]+FIELDS)
        writer.writeheader()
        for model in report["models"]:
            for r in model["rows"]:
                if r["true_label"] != r["pred_label"]:
                    values = {"model": model["name"], **{f:r[f] for f in FIELDS}}
                    # Avoid spreadsheet formula execution when someone opens the export.
                    writer.writerow({k: "'"+v if v.lstrip().startswith(("=", "+", "-", "@")) else v
                                     for k, v in values.items()})
    for i, model in enumerate(report["models"]):
        with (output / f"confusion_{i+1}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true_label / predicted_label"] + CLASSES)
            for label, row in zip(CLASSES, model["metrics"]["confusion_matrix"]):
                writer.writerow([label]+row)
    return output / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/local"))
    args = parser.parse_args()
    try:
        report = build_report(args.config)
        path = write_report(report, args.output)
    except (ValueError, KeyError, TypeError, OSError, csv.Error) as exc:
        parser.exit(2, f"Cannot build report: {exc}\n")
    print(report["caveat"])
    print(f"Saved offline report: {path}")


if __name__ == "__main__":
    main()
