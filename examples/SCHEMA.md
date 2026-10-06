# 预测 CSV 格式

编码 UTF-8；必须包含以下六列，额外列会被忽略。

| 字段 | 含义 |
|---|---|
| ID | 非空、在该文件中唯一的样本标识；对齐用，不读取图片 |
| true_label | 五个规定类名之一；须来自合法的验证真标签 |
| pred_label | 五个规定类名之一；当前方案的单标签预测 |
| fold | 该样本的验证折标识；跨方案必须一致 |
| split_id | 整个划分的标识；同文件只能一个值，跨方案必须一致 |
| source_kind | synthetic（合成夹具）或 validation（用户验证数据），必须与配置一致 |

类名大小写、空格和连字符严格匹配。请勿把未知测试标签或预测标签填为 true_label。

`demo_a.csv` 与 `demo_b.csv` 的全部 ID / 标签 / 预测是人工构造，模拟每类 5 个样本。它们不是训练输出，也不对应真实图片。

`from_oof.py` 接受原训练脚本的列：train_relpath、category、predicted_category、fold、covered；自动跳过 covered=False 的行。其 split ID 只证明样本标识/标签/折映射一致，不验证图像像素或泄漏。
