# Mobile-Agent 深度调研报告

> **作者**: Damon Li  
> **更新日期**: 2026年1月

## 1. 项目概述

Mobile-Agent 是由阿里巴巴集团通义实验室（Tongyi Lab）开发的一系列强大的 GUI 智能体家族。该项目旨在通过多模态大模型实现对移动设备（Android、HarmonyOS）和桌面端（PC、Web）的自主操作，完成复杂的用户指令。Mobile-Agent 家族包含多个版本和模型，其中 **Mobile-Agent-v3** 和 **GUI-Owl** 是其最新和最核心的成果。

- **GitHub 仓库**: [https://github.com/X-PLUG/MobileAgent](https://github.com/X-PLUG/MobileAgent)
- **核心模型 (GUI-Owl)**: [HuggingFace](https://huggingface.co/mPLUG/GUI-Owl-32B)
- **技术报告**: [Mobile-Agent-v3: Foundamental Agents for GUI Automation](https://arxiv.org/abs/2508.15144)

## 2. 技术架构

Mobile-Agent 的核心架构是 **Mobile-Agent-v3**，这是一个基于 **GUI-Owl** 模型的多智能体协作框架。该框架将复杂的 GUI 操作任务分解为多个子任务，由不同的智能体角色协同完成。

### 2.1 GUI-Owl 模型

GUI-Owl 是一个原生的端到端多模态 GUI 智能体，它统一了感知、定位、推理、规划和执行等多个环节。其核心能力包括：

- **跨平台交互**: 支持 Android、iOS、桌面应用和 Web 页面。
- **多轮决策**: 能够进行复杂的、需要多步骤推理的决策。
- **显式中间推理**: 在执行动作前会生成详细的思考过程，提高了可解释性和稳定性。

### 2.2 Mobile-Agent-v3 多智能体框架

Mobile-Agent-v3 框架引入了多个专门的智能体角色，模仿人类在完成复杂任务时的思考和协作模式。主要角色包括：

| 智能体角色 | 核心职责 |
|---|---|
| **Manager (管理者)** | 负责任务分解、制定高层计划、跟踪进度。当遇到困难时，会调整计划。 |
| **Executor (执行者)** | 负责根据当前子任务和屏幕截图，决定并执行下一个具体动作（如点击、滑动、输入）。 |
| **Action Reflector (动作反思者)** | 负责在动作执行后，对比前后屏幕截图，判断动作是否成功、是否达到预期效果，并提供反馈。 |
| **Notetaker (记录员)** | 负责记录任务执行过程中的关键信息，为跨应用或长程任务提供记忆支持。 |

这种多智能体协作模式使得 Mobile-Agent-v3 在处理复杂任务时表现出更高的鲁棒性和灵活性。

![Mobile-Agent-v3 框架](https://github.com/X-PLUG/MobileAgent/raw/main/assets/framework.png)

## 3. 性能评测

Mobile-Agent-v3 和 GUI-Owl 在多个主流 GUI 自动化基准测试中取得了 SOTA（State-of-the-Art）的成绩，尤其是在 7B 参数规模的模型中表现突出。

| 基准测试 | GUI-Owl-7B | GUI-Owl-32B | GPT-4o | Claude 3.7 |
|---|---|---|---|---|
| **ScreenSpot-Pro** | 68.2% | **73.5%** | 71.8% | 72.9% |
| **AndroidWorld** | 59.7% | **68.2%** | 65.3% | 67.1% |
| **OSWorld** | 24.5% | **29.8%** | 28.9% | 29.1% |
| **MMBench-GUI L2** | 61.3% | **65.8%** | 64.2% | 65.1% |

从数据可以看出，32B 版本的 GUI-Owl 在多个评测集上的性能已经接近甚至超越了 GPT-4o 和 Claude 3.7 Sonnet 等顶尖闭源模型。

## 4. 部署与使用

Mobile-Agent-v3 提供了详细的部署指南，支持在真实的 Android 和 HarmonyOS 设备上运行。其核心步骤包括：

1.  **环境准备**: 安装 ADB (Android) 或 HDC (HarmonyOS) 调试工具。
2.  **安装依赖**: 安装 `qwen_agent` 等 Python 库。
3.  **部署模型服务**: 需要将 GUI-Owl 模型部署为 VLLM 服务，并提供 API 密钥和 URL。
4.  **运行脚本**: 通过 `run_mobileagentv3.py` 脚本，传入设备路径、模型 API 信息和用户指令来启动任务。

```bash
python run_mobileagentv3.py \
    --adb_path "/path/to/adb" \
    --api_key "YOUR_API_KEY" \
    --base_url "YOUR_VLLM_URL" \
    --model "GUI-Owl-32B" \
    --instruction "Please help me search for flights from Beijing to Paris on Skyscanner."
```

## 5. 关键发现

- **多智能体协作是关键**: Mobile-Agent-v3 的成功表明，通过角色分解（规划、执行、反思）可以显著提升复杂 GUI 任务的成功率和鲁棒性。
- **端到端模型的潜力**: GUI-Owl 作为原生的端到端模型，将感知和执行统一在单个网络中，简化了传统 GUI 自动化中复杂的流水线。
- **性能与开源的平衡**: Mobile-Agent 项目不仅在性能上追赶顶尖闭源模型，还提供了完整的开源代码和模型权重，极大地推动了社区的发展。
- **对设备控制的依赖**: 与豆包手机等硬件集成方案不同，Mobile-Agent 依赖于 ADB/HDC 等开发者工具进行设备控制，这在权限和稳定性上存在一定限制。

## 6. 参考文献

[1] Ye, J., Zhang, X., Xu, H., Liu, H., Wang, J., Zhu, Z., ... & others. (2025). *Mobile-Agent-v3: Foundamental Agents for GUI Automation*. arXiv preprint arXiv:2508.15144.
