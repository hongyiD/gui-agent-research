# AppAgent 深度调研报告

> **作者**: Damon Li  
> **更新日期**: 2026年1月7日

## 一、项目概述

**AppAgent** 是腾讯开源的多模态智能体项目，全称为 "Multimodal Agents as Smartphone Users"（像智能手机用户一样的多模态智能体）。该项目已被 **CHI 2025** 会议接收，是 GUI Agent 领域的重要研究成果。

| 属性 | 信息 |
|------|------|
| **开发者** | 腾讯 QQ GY Lab |
| **论文** | [arXiv:2312.13771](https://arxiv.org/abs/2312.13771) |
| **开源协议** | MIT License |
| **GitHub** | [TencentQQGYLab/AppAgent](https://github.com/TencentQQGYLab/AppAgent) |
| **项目主页** | [appagent-official.github.io](https://appagent-official.github.io) |

![AppAgent Teaser](images/teaser.png)

## 二、核心创新

### 2.1 设计理念

AppAgent 的设计初衷是赋予 AI Agent **像人类用户一样的感知和操作能力**。它通过 ADB（Android Debug Bridge）获取手机屏幕截图，发送给多模态大模型。模型分析截图中的 UI 元素，决定下一步是点击哪个坐标还是滑动屏幕，从而实现真正的视觉交互。

### 2.2 两阶段学习机制

与其他开源项目不同，AppAgent **模仿了人类学习新软件的过程**：

```
┌─────────────────────────────────────────────────────────────┐
│                     探索阶段 (Exploration)                   │
│  ┌─────────────────────┐    ┌─────────────────────┐        │
│  │   自主探索模式       │    │   人类演示模式       │        │
│  │  (Autonomous)       │    │  (Human Demo)       │        │
│  └─────────────────────┘    └─────────────────────┘        │
│                          ↓                                  │
│              生成 UI 元素文档 (Knowledge Base)               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                     部署阶段 (Deployment)                    │
│  - 接收用户任务                                              │
│  - 调用知识库中的文档                                        │
│  - 精准执行操作                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 知识库 (Knowledge Base)

当给 AppAgent 一个新的 App 时，它可以通过：
1. **自主探索** - 自动尝试各种操作并记录结果
2. **观察人类演示** - 学习人类的操作方式

在这个过程中，它会记录下每个按钮是干什么的，并生成一份使用文档。当需要执行具体任务时，它会调用之前生成的文档，精准地执行操作，而不需要每次都重新摸索。

## 三、技术架构

### 3.1 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户任务                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                     AppAgent 核心                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 截图 + UI 元素解析                    │   │
│  │  - 获取屏幕截图                                       │   │
│  │  - 解析 XML 获取可点击/可聚焦元素                     │   │
│  │  - 为元素添加数字标签                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   多模态模型推理                      │   │
│  │  - GPT-4V / Qwen-VL                                  │   │
│  │  - 分析截图 + 任务描述 + UI 文档                      │   │
│  │  - 输出下一步动作                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    动作执行                           │   │
│  │  - tap / text / long_press / swipe                   │   │
│  │  - 通过 ADB 控制设备                                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| **任务执行器** | `task_executor.py` | 执行用户任务的主循环 |
| **自主探索器** | `self_explorer.py` | 自主探索 App 并生成文档 |
| **人类演示记录** | `step_recorder.py` | 记录人类操作演示 |
| **文档生成器** | `document_generation.py` | 生成 UI 元素文档 |
| **Android 控制器** | `and_controller.py` | ADB 设备控制 |
| **模型接口** | `model.py` | GPT-4V / Qwen-VL 接口 |
| **提示词模板** | `prompts.py` | 各阶段的提示词 |

## 四、动作空间

AppAgent 定义了简化的动作空间，模拟人类交互：

| 动作 | 参数 | 描述 |
|------|------|------|
| `tap` | `area` | 点击指定数字标签的元素 |
| `text` | `input_str` | 输入文本 |
| `long_press` | `area` | 长按指定元素 |
| `swipe` | `area`, `direction`, `distance` | 在指定元素上滑动 |
| `grid` | - | 启用网格模式 |
| `tap_grid` | `area`, `subarea` | 网格模式下点击 |
| `swipe_grid` | `start`, `end` | 网格模式下滑动 |
| `FINISH` | - | 任务完成 |

### 网格模式

当界面上没有合适的数字标签时，AppAgent 可以启用网格模式，将屏幕划分为多个区域，允许点击任意位置：

```
┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │
├───┼───┼───┼───┤
│ 5 │ 6 │ 7 │ 8 │
├───┼───┼───┼───┤
│ 9 │10 │11 │12 │
└───┴───┴───┴───┘
```

每个区域还可以细分为 9 个子区域（top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right）。

## 五、UI 文档结构

每个 UI 元素的文档包含以下字段：

```python
doc_content = {
    "tap": "",        # 点击该元素的功能描述
    "text": "",       # 文本输入的用途描述
    "v_swipe": "",    # 垂直滑动的功能描述
    "h_swipe": "",    # 水平滑动的功能描述
    "long_press": ""  # 长按的功能描述
}
```

示例：
```python
{
    "tap": "Tapping this button opens the user's profile page.",
    "text": "This field is used to enter the search query.",
    "v_swipe": "Swiping up scrolls through the feed.",
    "h_swipe": "",
    "long_press": "Long pressing shows additional options menu."
}
```

## 六、自主探索机制

### 6.1 探索流程

```
1. 截取当前屏幕截图
2. 解析 UI 元素树，获取可交互元素
3. 为元素添加数字标签
4. 调用 LLM 决定下一步动作
5. 执行动作
6. 截取动作后的截图
7. 调用 LLM 反思动作效果
8. 根据反思结果生成/更新文档
9. 重复直到任务完成或达到最大轮数
```

### 6.2 反思机制

每次动作后，AppAgent 会进行反思，判断动作效果：

| 判断结果 | 含义 | 后续处理 |
|----------|------|----------|
| `INEFFECTIVE` | 动作无效 | 将元素加入无用列表 |
| `BACK` | 需要返回 | 执行返回操作 |
| `CONTINUE` | 继续探索 | 记录文档并继续 |
| `SUCCESS` | 动作成功 | 记录文档 |

## 七、支持的模型

| 模型 | 提供商 | 性能 | 费用 |
|------|--------|------|------|
| `gpt-4-vision-preview` | OpenAI | 最佳 | ~$0.03/请求 |
| `qwen-vl-max` | 阿里云 | 较差 | 免费 |

## 八、安装与使用

### 8.1 环境准备

```bash
# 1. 安装 ADB
# 下载 Android SDK Platform Tools

# 2. 连接设备
adb devices

# 3. 安装依赖
pip install -r requirements.txt
```

### 8.2 配置

编辑 `config.yaml`：

```yaml
MODEL: OpenAI  # 或 Qwen
OPENAI_API_KEY: your_api_key
OPENAI_API_MODEL: gpt-4-vision-preview
REQUEST_INTERVAL: 3  # 请求间隔（秒）
MAX_ROUNDS: 20       # 最大轮数
```

### 8.3 探索阶段

```bash
python learn.py
# 选择探索模式：
# 1. Autonomous exploration（自主探索）
# 2. Human demonstration（人类演示）
```

### 8.4 部署阶段

```bash
python run.py
# 输入应用名称和任务描述
```

## 九、AppAgentX

2025年3月，团队发布了 **AppAgentX**，下一代 GUI Agent，具有进化机制（evolving mechanism）。

项目主页：[appagentx.github.io](https://appagentx.github.io/)

## 十、优势与劣势

### 优势

1. **知识积累**: 通过探索阶段积累应用知识，提高后续任务成功率
2. **人类演示**: 支持从人类演示中学习，降低探索成本
3. **网格模式**: 可以点击任意屏幕位置，不受 UI 元素限制
4. **反思机制**: 自动评估动作效果，生成高质量文档
5. **MIT 协议**: 商业友好

### 劣势

1. **无自有模型**: 依赖 GPT-4V 或 Qwen-VL，无专门训练的 GUI 模型
2. **探索成本**: 每个新应用都需要探索阶段
3. **API 费用**: GPT-4V 调用费用较高
4. **仅支持 Android**: 不支持 iOS

## 十一、论文引用

```bibtex
@misc{yang2023appagent,
      title={AppAgent: Multimodal Agents as Smartphone Users}, 
      author={Chi Zhang and Zhao Yang and Jiaxuan Liu and Yucheng Han and Xin Chen and Zebiao Huang and Bin Fu and Gang Yu},
      year={2023},
      eprint={2312.13771},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 参考资料

1. [AppAgent GitHub Repository](https://github.com/TencentQQGYLab/AppAgent)
2. [AppAgent Paper (arXiv)](https://arxiv.org/abs/2312.13771)
3. [AppAgent Project Page](https://appagent-official.github.io)
4. [AppAgentX](https://appagentx.github.io/)
