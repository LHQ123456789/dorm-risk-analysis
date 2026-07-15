# 宿舍安全风险识别系统

> 一个基于视觉语言模型（VLM）的宿舍安全隐患自动识别工具

---

## 📖 关于本项目

这是我在大学**创新实践课程**中的一个尝试性项目。在课程期间，我第一次接触视觉语言模型（VLM）这个领域，出于兴趣，决定做一个看起来"有点用"的小工具——让 AI 帮忙检查宿舍里的安全隐患。

由于**本地设备性能有限**（一台 RTX 3060 笔记本，显存只有 6GB），大部分实验和正式运行都是在 **AutoDL 云端**（RTX 5090）完成的。整个过程踩了不少坑，但也因此对 VLM 的 Prompt 工程、模型量化、推理部署有了初步的实践认知。

---

## 🔧 做了什么

- 定义了 **6 类宿舍安全风险**：通道阻塞、触电风险、火灾风险、坠物风险、危险物暴露、清洁度异常
- 基于 **Qwen3-VL-8B-Instruct** 视觉语言模型，实现自动识别
- 经历了 **4 个版本的 Prompt 迭代**（V1 Baseline → V2 CoT → V3 Full → V3 宽松版）
- 输出**结构化 JSON + Markdown 风险报告**
- 支持**本地 / 云端双环境**运行

---

## 🧠 学到了什么

- 视觉语言模型的基本原理和使用方式
- Prompt 工程对模型输出的巨大影响（同一个模型，Prompt 不同，结果天差地别）
- 模型规模对复杂指令执行能力的关键差异（2B vs 8B）
- 云端 GPU 环境的基本操作（AutoDL、ModelScope、HuggingFace）
- 结构化输出的鲁棒解析（截断 JSON 修复、多级回退提取）

---

## 🚀 快速开始

### 环境要求

| 方案 | GPU | 显存 | 模型 |
|------|-----|------|------|
| 云端（推荐） | RTX 4090 / 5090 | ≥ 24 GB | Qwen3-VL-8B-Instruct |
| 本地 | RTX 3060+ | ≥ 6 GB | Qwen3-VL-2B-Instruct |

### 安装依赖

```bash
pip install torch transformers accelerate pillow modelscope
```

### 下载模型 & 运行

```bash
# 下载模型（从 ModelScope）
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-VL-2B-Instruct', cache_dir='./models')"

# 运行分析
python risk_analyser.py -i ./dorm_photos/ -o results.json -r risk_report.md
```


---

---

## 📝 局限与不足

- 测试数据有限，缺乏大规模标注数据集，无法计算精确的召回率/准确率
- 目前仅针对宿舍场景设计，泛化到其他场景需要调整 Prompt
- 2B 模型下 JSON 解析不够稳定，复杂 Prompt 执行效果不佳
- 缺乏多模型横向对比（计划中）

---

## 🔭 未来计划

希望后续能将这个项目**持续优化**，并最终**部署到机器人平台上**，实现真正自主的室内安全巡检——不只是"看图片"，而是让机器人边走边看，实时发现风险并告警。为此需要：

- 构建标注数据集，进行定量评估
- 探索模型微调（LoRA / QLoRA），提升小模型表现
- 尝试 ONNX / TensorRT 量化部署，适配边缘设备
- 接入机器人视觉管线，实现实时巡检

---



*一个初学者的尝试，欢迎交流与指正。*
