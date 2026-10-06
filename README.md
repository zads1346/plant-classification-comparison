# 植物分类实验与结果对比

一个面向五类植物图像分类的**实验复现与结果对比项目**。保留已有的训练 Notebook / Python 实现，并增加一个轻量、离线运行的结果对比工具，集中查看同一验证集上的指标、混淆矩阵和错误样本。

> 当前为公开源码预览版。页面中的方案 A/B 来自 25 条人工构造的合成预测，**不是 EfficientNet、ConvNeXt 或真实比赛成绩**。本次没有训练新模型、没有生成真实提交文件，也没有新的提分结论。

## 先预览

1. 在 GitHub 仓库选择 **Code → Download ZIP** 并解压，或使用 Git 克隆本仓库。
2. 用浏览器直接打开 `preview/index.html`，无需服务器、联网、GPU 或安装前端依赖。
3. 切换方案查看混淆矩阵；按 ID / 类别筛选错例；打开“项目怎么打包”查看项目介绍。

也可以只下载 `preview/index.html`，保存后用浏览器打开。某些聊天应用不会直接执行 HTML，下载到电脑后打开即可。

## 已有、这次新增、尚未做

| 状态 | 内容 |
|---|---|
| 已有代码 | `train/plant2026_improved.py` 与独立 Notebook；数据审计、分层/分组折划分、训练、OOF、提交 CSV、错例与指标输出 |
| 本次新增并可运行 | 读取预测 CSV、检查同一批样本与真实标签及折划分、计算指标、导出本地 HTML/JSON/CSV、方案切换和错例筛选、从已有 OOF 输出转换 |
| 目前只做演示 | 25 条合成样本，5 类各 5 条；方案 A 15 条正确，方案 B 18 条正确；这只用于检查软件计算与展示 |
| 尚未实现 | 在网页上传图片识别、错例原图展示、自动批量训练/任务调度、训练曲线集成、EXE 桌面应用 |
| 尚未验证 | 真实 GPU 训练、真实数据完整流程、新版模型成绩、耗时、显存占用、泛化表现 |

## 目录

```text
plant-classification-comparison/
├── README.md
├── train/                 # 原有独立脚本 + Notebook（保留实现）
├── eval/                  # compare.py；from_oof.py
├── configs/               # demo.json；real.example.json
├── examples/              # 合成演示 CSV；格式说明
├── preview/               # index.html；报告模板
├── reports/               # 本地生成结果，默认不提交
├── tests/                 # 标准库单元测试
├── docs/                  # 实际验证范围、原说明和依赖边界
├── requirements-preview.txt
├── requirements-selftest.txt
├── .gitignore
└── LICENSE-NOTICE.md       # 许可证待本人决定
```

## 运行离线报告工具

在解压后的项目根目录运行。Python 3.10+；本次在 Python 3.12.14 上验证。此步骤只使用标准库，不需 `pip install`，也不会下载模型或数据。

```bash
python eval/compare.py --config configs/demo.json --output reports/demo
python -m unittest discover -s tests -v
```

Windows 如只有 `py` 命令，将上面的 `python` 换成 `py -3`。生成后打开 `reports/demo/index.html`。

输出：
- `index.html`：自包含离线对比报告
- `metrics.json`：准确率、micro-F1、macro-F1、逐类指标、CSV 的 SHA-256
- `errors.csv`：各方案错例
- `confusion_1.csv`、`confusion_2.csv` 等：行真实 / 列预测的混淆矩阵

演示结果明确标记 `source_kind=synthetic`。把演示配置改成 validation 但仍使用合成 CSV 时，工具会拒绝，防止误标。

## 接入自己的真实验证结果

只接受**有真实标签的验证/OOF 预测**。`submission.csv` 的 `ID,Category` 不包含测试真标签，不能计算真实测试准确率，不要补造标签。

已有训练脚本会在实际成功结束后生成 `oof_predictions.csv`。若有两个真实运行目录，可执行：

```bash
python eval/from_oof.py --input runs/run_a/oof_predictions.csv --output examples/real_a.csv
python eval/from_oof.py --input runs/run_b/oof_predictions.csv --output examples/real_b.csv
python eval/compare.py --config configs/real.example.json --output reports/real
```

上面 `run_a` / `run_b` 是你实际运行目录的占位名称，本包没有这些真实结果。先修改 `configs/real.example.json` 的方案名、说明、验证范围和文件路径。配置文件中的 CSV 路径相对该配置文件所在目录解释。

转换器只取 `covered=True` 行，并根据完整样本 ID、真实标签、折分配生成 split ID。对比工具强制检查两方案的样本集合、真实标签、fold、split ID 完全一致，不要求 CSV 行顺序相同。一个单折和一个完整五折结果不能直接比较。

这些检查**不能**证明图像像素相同、标签来源可信或模型从未见过验证数据。做正式实验时仍需核对原训练清单、图像哈希、分组、配置、种子和训练隔离；多次依据同一验证集选择模型也会带来选择偏差。

## 数据与指标约定

- 五类：Black-grass、Common wheat、Loose Silky-bent、Scentless Mayweed、Sugar beet。
- 既有任务背景：500 张有标签训练图、378 张无标签测试图，见原说明。这是此前记录，本次预览没有重新读取竞赛数据。
- 本任务每张图只有一个类别，且每张图给出一个完整类别预测：micro-F1 = accuracy。
- macro-F1 是 5 个固定类别 F1 的平均，缺失类别按 0 处理；用于看类别差异，不拿它冒充比赛主指标。
- 混淆矩阵行是真实类别，列是预测类别。示例“另一方案正确”仅是逐个错例对照，不意味着统计显著改进。
- 报告没有伪造置信度、耗时、显存、排行榜或真实训练指标。

详细 CSV 字段见 `examples/SCHEMA.md`。

## 已有训练代码如何使用

`train/` 来自此前生成的改进版，本次未改训练算法。Notebook 自包含实现，不需要旁边的 `.py` 文件。

- fast：EfficientNet-B2，五折划分中的一个折；属于 partial OOF / 单个留出验证。
- full：EfficientNet-B2 + ConvNeXt-Tiny，各五折，总计十个模型。
- 两者都保留各折结果，报告完整覆盖范围；验证与测试用同一推理函数。
- 默认 `pretrained_allowed=False`（Notebook 为 `PRETRAINED_ALLOWED=False`）。这会在下载权重和训练前停止，直到课程/主办方许可被确认。这个停止是保护检查，不等于已经训练失败或完成。

可以先在有依赖的环境中跑非训练自检：

```bash
python train/plant2026_improved.py --self-test
```

此自检需要 numpy、pandas、Pillow、scikit-learn，不需 PyTorch 或 GPU。若新建本地独立环境，可安装 `requirements-selftest.txt` 中本次实测版本；它不是 GPU 训练环境锁文件，也不应拿来盲目覆盖 Kaggle 自带环境。

正式训练建议按已有 Notebook 的说明，在自己的 Kaggle 环境中导入 `train/plant2026_improved.ipynb`，附加有权使用的数据，选择 GPU，检查依赖，再逐格执行。路径不同时调整配置。**仅在本人确认课程允许 ImageNet 预训练之后**，才启用该开关。

脚本确实支持以下等价命令；这是使用说明，本次没有执行：

```bash
python train/plant2026_improved.py --preset fast --allow-pretrained
python train/plant2026_improved.py --preset full --allow-pretrained
```

也支持 `--data-root`、`--template-path`、`--output-dir`、`--weights-dir`、`--groups-csv` 等参数。若禁止预训练，`--scratch` 使用随机初始化，但需要另调学习率与训练预算，不能保证效果等同迁移学习。训练所需的 torch / torchvision 配套 GPU 环境本次未建立或验证，详见 `docs/DEPENDENCIES.md`。

## 为什么先给源码包，不先做 EXE

GitHub 项目首先需要可读源码、说明、配置示例、测试和可复现步骤。训练代码依赖 GPU / 数据 / 运行环境，把它塞进 EXE 不能消除这些前提。当前离线 HTML 已能满足“先看工具效果”；未来真的需要非技术用户双击运行时，再决定是否做桌面界面或 Web 应用。

推荐仓库描述：

> 五类植物图像分类的训练复现与实验结果对比工具，支持同一验证集上的 micro-F1 / macro-F1、混淆矩阵和错例分析，并提供离线 HTML 报告。

推荐在展示中说明：

> 训练部分已有实现；新增对比层已用合成数据通过指标计算测试与交互脚本语法检查，尚未完成真实浏览器界面和点击验收。真实模型性能仍以本人合法数据上的实际运行记录为准，当前页面不构成提分证明。

## 发布与数据边界

- 仓库所有者已授权公开本项目；本仓库不替代课程对公开代码、预训练权重和数据使用的政策要求，相关许可仍需本人核对。
- 不上传真实数据集、模型权重、测试标签、账号信息、API 密钥或私人路径。
- `.gitignore` 是辅助，不能代替逐个审核；已被 Git 跟踪的文件不会因新增忽略规则自动消失。
- 本包不含数据集或模型权重。`examples/` 只有合成 CSV；Notebook 没有执行输出。
- 许可证尚未决定，见 `LICENSE-NOTICE.md`；不要未经确认宣称 MIT 开源。
- 本仓库发布源码与合成预览；发布不代表已完成真实浏览器验收、比赛提交或新训练。

实际测试范围见 `docs/VALIDATION.md`；旧版历史说明保留在 `docs/original_README_zh.txt`，其中历史日志不是本次训练结果。
