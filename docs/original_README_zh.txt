2026task1 植物分类：完整改进版
生成日期：2026-10-05

先说结论
这份代码优先修复原 Notebook 中已经能从日志证明的问题：选模型时使用的验证方法，与实际生成提交时的方法不同。它没有新的实测比赛分数，也不保证一定提分。原文件未被覆盖。

文件
1. plant2026_improved.ipynb：可导入 Kaggle 的独立完整 Notebook，不依赖旁边的 Python 文件。
2. plant2026_improved.py：同一套实现的独立 Python 脚本。
3. README_zh.txt：运行方法、实际诊断与实验建议。
4. VALIDATION.txt：已完成与尚未完成的检查。
5. self_test.log：合成数据检查输出，不是比赛训练结果。
ZIP 包含上述文件；不包含用户原 Notebook、数据集、模型权重或伪造的 submission.csv。

一、你当前 Notebook 的真实情况
证据来源是上传的 task01 (1).ipynb 的第 3、4 个代码单元及其保留输出。

- 训练日志显示 Device: cuda，GPU: Tesla T4。虽然 Notebook 元数据中的 isGpuEnabled 为 false，但不能据此断言原运行没用 GPU。
- 训练集 500 张，每类 100 张；测试集和模板都是 378 行，原 Notebook 已验证 ID 集合完全一致。
- 五个类名为 Black-grass、Common wheat、Loose Silky-bent、Scentless Mayweed、Sugar beet。原有类别映射和按模板顺序输出的实现是正确的，新版保留并增加检查。
- 原运行耗时日志为 46.2 分钟。新版耗时未实测，不能承诺某个分钟数。

最关键的验证差异：
- EfficientNet-B2：各折保存时单图 accuracy 均值 0.906；后来换成裁剪 + TTA 的 OOF accuracy 为 0.902。
- ConvNeXt-Tiny：0.922 → 0.898。
- ResNet50：0.904 → 0.886。
这些折的验证样本数相等，因此平均 accuracy 与合并 OOF accuracy 可比较。它证明这套“后加的裁剪和 TTA 组合”在这些检查点上产生退步；尚不能把全部损失归因于裁剪或某一种变换。

另外几个确定的问题：
1. 测试预测额外加入 20% 的 320px 高分辨率分支，OOF 没有。因此所报 OOF 分数并不是实际提交配方的验证分数。
2. 七个 TTA 视图中，“同时翻转高和宽”与“旋转 180 度”重复，实际只有六种不同变换。新版 D4 选项是八种唯一变换，但默认不用 TTA。
3. 原 Stage 2 使用 185 张伪标签测试图，在全部有标签数据上训练后，未在独立留出集上验证就混入最终结果 30%。0.906 的 teacher OOF 不能作为这份最终提交的验证分数。
4. 所谓伪标签 precision=1.0000 来自在同一批 OOF 标签上搜阈值并报告成绩，不是测试伪标签必然正确的证明。
5. 类别偏置搜索打印 macro-F1=0.9098，但算出的 calibrated 概率没有进入 final_probs；该值不是最终 CSV 的成绩。
6. 原代码删去了每种架构四个折模型，使后续低成本对比验证很困难。
7. 原 MixUp 下的 train accuracy 把混合图片的预测和原标签直接比较，不能用它简单判断是否欠拟合。

不是问题的地方：
- 比赛评估页正文名为 MeanF1Score，但公式图明确写的是 F1_micro。单标签、多类别且每张图只预测一个完整类别时，micro-F1 等于 accuracy。因此原 CHECKPOINT_METRIC="accuracy" 本身是正确的，不应盲目改成 macro-F1。
- 不应该为了训练集每类 100 张，就强迫测试集预测数量相等。测试分布可能不同。
- 单独成绩较低的模型仍可能给集成带来互补收益，不能仅凭单模分数断言 ResNet 没用。新版两架构只是节省实验成本的预设，并非已证明最佳的组合。

二、新版具体做了什么
- 默认 EfficientNet-B2 单折 fast；full 为 EfficientNet-B2 + ConvNeXt-Tiny，五折，共十个模型。
- 主指标 micro-F1，同时输出 accuracy、macro-F1、NLL、逐类报告和混淆矩阵。
- 逐轮验证、最佳检查点复测、OOF 和测试预测使用同一个 predict()；尺寸、归一化、TTA 一致。
- 默认保留原单图验证使用的整图拉伸到 288×288，移除绿植阈值裁剪与随机裁剪，不丢掉整株信息。letterbox 可保持比例，但作为可选实验，没有宣称它已经提分。
- 温和颜色变化、翻转、旋转；旋转扩展画布，避免额外切掉边缘。训练准确率不再受 MixUp 标签含义影响。
- ImageNet 的 RGB 均值/标准差保持正确。图片读取处理 EXIF 方向和透明背景。
- 使用可选 warm-up EMA，默认开启；分组学习率、短头部训练、骨干解冻、余弦学习率、AMP、梯度裁剪、早停。
- 预训练模型默认冻结 BatchNorm 的运行统计量，减少小数据/小批次漂移。这是可调整的实验设置，不是已证明的根因。
- 完全相同的解码图像放在同一折，冲突标签直接报错。可选 groups_csv 把同一植株/拍摄批次放在同组。
- 不自动识别所有近重复图片；如同一植株有多个拍摄角度，需要你提供组信息。单纯随机分层并不能排除这类泄漏。
- 不使用测试伪标签，不搜索 OOF 集成权重或类别偏置；固定等权集成，不把调参收益当成独立测试收益。
- 保留每折 best.pt、逐轮 history.csv、OOF/测试概率、原始划分和运行配置。
- 完成的折可安全复用。配置、输入像素/标签/折分配、依赖版本和本地预训练文件的指纹不匹配时停止，不会悄悄混用旧结果。
- 未完成的折重新训练，不支持从某个未完成 epoch 精确续训。

三、Kaggle 运行步骤
1. 新建或导入 plant2026_improved.ipynb，附加 2026task1 比赛数据。不要覆盖旧版本。
2. Notebook 设置里选择 GPU，例如 Tesla T4。不要只相信导入的元数据；运行环境单元会打印实际 CUDA 状态。
3. 首次运行先执行定义单元、self_test()。可选执行 torch_smoke_test()，它只跑合成图和微型 CNN，不下载权重，不能用来估算比赛分数。
4. 先确认课程/主办方允许 ImageNet 预训练权重。2026-10-05 看到的公开规则要求深度学习，但没有明确说明是否允许预训练、外部数据或伪标签。新版默认“待确认”并主动阻止训练，不会偷偷下载。确认允许后，配置单元把 PRETRAINED_ALLOWED = True。
   - 如果禁止预训练：把 USE_PRETRAINED=False。代码能走从零训练，但 500 张图上的效果不等同迁移学习，这套学习率/轮数并未针对从零训练调优。
   - 只有允许下载权重时才开启 Internet；也可将官方 torchvision 权重附加为 Notebook 输入，设置 WEIGHTS_DIR。
   - weights_dir 需要对应官方文件名：efficientnet_b2_rwightman-c35c1473.pth，以及 full 模式的 convnext_tiny-983f1562.pth。不要下载陌生来源的 pickle 模型。
5. 先 PRESET="fast" 跑通一个分层留出折。它约用 400 张训练、100 张验证，仅覆盖 1/5 训练样本。报告会标为 partial OOF，绝不称为完整五折成绩。
6. 确认输入、日志与输出正常后，切为 PRESET="full"，使用不同输出目录。full 每张训练图都获得一次折外预测。
7. 提交 run() 打印的 submission.csv。fast 默认路径：/kaggle/working/plant2026_improved_fast/submission.csv；full 为 /kaggle/working/plant2026_improved_full/submission.csv。CSV 只有 ID,Category，保留模板顺序。
8. 新版不自动向 Kaggle 提交。不要把 synthetic_submission.csv 或任何测试夹具当作真实提交。

命令行等价操作（已确认预训练允许时）：
python plant2026_improved.py --self-test
python plant2026_improved.py --torch-smoke-test
python plant2026_improved.py --preset fast --allow-pretrained
python plant2026_improved.py --preset full --allow-pretrained

路径不同时使用 --data-root、--template-path、--output-dir。不需要 Kaggle API token。
Notebook 已把全套实现内嵌，不需要 %run 外部文件或 !pip install。

四、怎么做有意义的下一轮实验
先固定一个划分、一个种子、一个模型和同一套验证流程。不要一边改变 TTA、尺寸、模型、伪标签，一边把不同含义的分数直接比较。

建议顺序：
A. 单图 288px 基线：tta="none", resize_mode="stretch"。
B. 仅把 tta 改为 "hflip"；验证/测试一起改。然后再考虑 "d4"。
C. 单独试 resize_mode="letterbox"；保持其他配置相同。
D. 单独比较 EMA、BatchNorm 冻结或 320px；变尺寸时验证和测试同时变化。
E. 预算允许再扩大到 full 五折，并查看两模型的错误是否互补。

每次改配置都使用新的 output_dir。不要根据测试集预测分布、人眼猜测测试类别或公开榜的小幅波动反复调参。代码不包含任何外部测试真标签。
早停本身使用了验证集，因此 OOF 也不是完全未经选择的最终泛化估计；反复比较很多版本仍会过拟合 CV。需要更严格估计时再做独立留出集或嵌套验证。

如显存不足：减少 batch_size（例如 16→8），保持评估配方不变，并换输出目录。workers 出现进程问题可改为 0。模型加载失败不会自动改成随机初始化。
代码面向 Kaggle 的 PyTorch 2.x/torchvision 配套环境，使用 torch.amp.GradScaler 与 weights_only=True。不要为了消除报错盲目覆盖 Kaggle 自带的 torch/torchvision。

五、可核查的来源
原始日志：用户上传 task01 (1).ipynb，运行时间戳 2026-09-27，第四代码单元。
官方评估公式（需看图，纯文本抓取会漏掉 micro 下标）：
https://www.kaggle.com/competitions/2026task1/overview/evaluation
比赛规则：
https://www.kaggle.com/competitions/2026task1/rules
比赛数据页面：
https://www.kaggle.com/competitions/2026task1/data
PyTorch 官方 AMP 示例：
https://docs.pytorch.org/docs/2.14/notes/amp_examples.html
Torchvision EfficientNet-B2：
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html
Torchvision ConvNeXt-Tiny：
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html
scikit-learn 分组分层交叉验证：
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
scikit-learn F1 定义：
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

最终边界
这次拿到的是代码与旧输出，没有拿到真实图片、当前模型检查点或新的 Kaggle 运行权限。已完成代码/数据流程检查；真实网络训练、显存占用、训练速度和比赛分数仍待在 Kaggle 上验证。没有给出“95%+”等未经实测的保证。
