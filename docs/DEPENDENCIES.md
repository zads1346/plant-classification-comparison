# 依赖与已验证范围

## 离线预览 / 对比工具

- 直接打开 HTML：浏览器即可，报告内嵌数据、样式和脚本，没有 CDN、网络请求或前端构建依赖。
- 重新生成报告、转换 OOF、运行单元测试：Python 3.10+ 标准库，本次在 Python 3.12.14 实测。
- 没有要求安装 Flask、Streamlit、React、PyTorch 或 pandas 才能看报告。

## 训练脚本的非训练自检

根据已有源码导入确认依赖：numpy、pandas、Pillow、scikit-learn。
本次实际环境：numpy 2.3.5、pandas 2.2.3、Pillow 12.3.0、scikit-learn 1.8.0。
`requirements-selftest.txt` 只记录这些实测版本；自检使用临时合成图片，不训练网络。

## 真实训练

已有源码另需相互兼容的 torch / torchvision，并默认要求 CUDA GPU。源码使用官方 torchvision 的 efficientnet_b2 和 convnext_tiny、AMP、`weights_only=True` 等 API。

本次环境没有安装 torch / torchvision，因此没有伪造全量 GPU 依赖锁文件、训练通过记录或具体 CUDA 组合。请在课程认可的 GPU 环境里检查配套版本，沿用该环境的正确安装，不要仅为打开预览而安装训练依赖。

官方安装与模型文档（供运行者自行核对，未据此宣称本次训练通过）：
- https://pytorch.org/get-started/locally/
- https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html
- https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html

数据和预训练权重不在包中。启用官方权重前须先核实课程许可，下载行为不属于本次预览。
