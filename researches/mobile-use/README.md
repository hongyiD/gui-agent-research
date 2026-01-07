# mobile-use 深度调研报告

> **作者**: Damon Li  
> **更新日期**: 2026年1月7日

## 一、项目概述

### 1.1 基本信息

| 属性 | 值 |
|------|-----|
| **项目名称** | mobile-use |
| **开发者** | Minitap AI |
| **开源地址** | https://github.com/minitap-ai/mobile-use |
| **Star 数** | 1.8k+ |
| **许可证** | MIT |
| **支持平台** | Android / iOS |

### 1.2 项目定位

mobile-use 是一个由 Minitap AI 团队开源的 Python 库，旨在通过自然语言控制手机。它的核心理念是：

> **"Say something, automate your phone"** - 说句话，自动化你的手机

![mobile-use Banner](images/banner-v2.png)

### 1.3 核心特性

| 特性 | 描述 |
|------|------|
| **自然语言控制** | 使用自然语言与手机交互 |
| **UI 感知自动化** | 智能导航应用界面（游戏除外，因为游戏不提供 Accessibility Tree） |
| **数据抓取** | 从任何应用提取信息并结构化为指定格式（如 JSON） |
| **可扩展可定制** | 支持配置不同的 LLM 来驱动智能体 |

## 二、技术架构

### 2.1 多智能体系统

mobile-use 采用了复杂的多智能体协作架构，包含以下核心智能体：

![mobile-use 架构图](images/graph.png)

| 智能体 | 职责 |
|--------|------|
| **Planner** | 任务规划器，将用户目标分解为子目标 |
| **Orchestrator** | 编排器，管理子目标的执行顺序和状态 |
| **Cortex** | 大脑，分析屏幕状态并做出决策 |
| **Executor** | 执行器，将决策转换为具体的设备操作 |
| **Contextor** | 上下文管理器，维护任务执行的上下文信息 |
| **Summarizer** | 总结器，汇总执行结果 |
| **Outputter** | 输出器，格式化最终输出 |
| **Hopper** | 跳转器，处理应用间的切换 |

### 2.2 工作流程

```
用户输入 → Planner → Orchestrator → Cortex → Executor → 设备
                ↑                                    │
                └────────── 反馈循环 ─────────────────┘
```

1. **用户输入**: 自然语言描述任务目标
2. **Planner**: 将目标分解为可执行的子目标列表
3. **Orchestrator**: 管理子目标的执行状态，决定下一步
4. **Cortex**: 分析当前屏幕截图和 UI 层次结构，做出操作决策
5. **Executor**: 执行具体的点击、滑动、输入等操作
6. **反馈循环**: 根据执行结果更新状态，继续下一轮

### 2.3 设备控制层

mobile-use 支持多种设备连接方式：

| 平台 | 连接方式 | 实现 |
|------|----------|------|
| **Android 物理设备** | USB + ADB | `UIAutomatorClient` |
| **Android 模拟器** | ADB | `UIAutomatorClient` |
| **iOS 模拟器** | fb-idb | `IdbClientWrapper` |
| **iOS 物理设备** | WebDriverAgent | `WdaClientWrapper` |
| **BrowserStack** | 云服务 | `BrowserStackClientWrapper` |

## 三、核心技术

### 3.1 LLM 集成

mobile-use 使用 LangChain 框架集成多种 LLM：

```python
# 支持的模型提供商
- OpenAI (GPT-4, GPT-4V)
- Google Vertex AI (Gemini)
- 任何 OpenAI API 兼容的本地模型
```

每个智能体可以配置不同的 LLM：

```jsonc
// llm-config.override.jsonc
{
  "planner": {
    "provider": "openai",
    "model": "gpt-4o"
  },
  "cortex": {
    "provider": "google_vertex",
    "model": "gemini-2.0-flash"
  }
}
```

### 3.2 UI 理解

mobile-use 结合多种信息源理解 UI：

1. **屏幕截图**: 视觉信息，用于 VLM 分析
2. **UI 层次结构**: Accessibility Tree，提供元素的结构化信息
3. **焦点应用信息**: 当前前台应用的包名和活动名

### 3.3 动作空间

支持的设备操作：

| 动作 | 描述 |
|------|------|
| `tap(x, y)` | 点击指定坐标 |
| `long_press(x, y)` | 长按指定坐标 |
| `swipe(x1, y1, x2, y2)` | 从起点滑动到终点 |
| `type_text(text)` | 输入文本 |
| `press_key(key)` | 按下按键（Home、Back 等） |
| `scroll(direction)` | 滚动屏幕 |
| `open_app(package)` | 打开指定应用 |

## 四、性能评测

### 4.1 AndroidWorld 基准

mobile-use 在 AndroidWorld 基准测试中表现优异：

![Benchmark](https://files.peachworlds.com/website/753680a0-c383-451e-9800-703d04489ea9/comparison.png)

> 来源: https://minitap.ai/benchmark

### 4.2 官方排行榜

mobile-use 位列 AndroidWorld 官方排行榜前列：
- [AndroidWorld Leaderboard](https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/edit?pli=1&gid=0#gid=0)

## 五、部署指南

### 5.1 快速启动（Docker）

```bash
# Linux/macOS
chmod +x mobile-use.sh
bash ./mobile-use.sh \
  "Open Gmail, find first 3 unread emails, and list their sender and subject line" \
  --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
```

### 5.2 手动安装

```bash
# 1. 克隆仓库
git clone https://github.com/minitap-ai/mobile-use.git && cd mobile-use

# 2. 设置环境变量
cp .env.example .env
# 编辑 .env 添加 API 密钥

# 3. 创建虚拟环境
uv venv
source .venv/bin/activate  # Linux/macOS

# 4. 安装依赖
uv sync
```

### 5.3 运行示例

```bash
# 基本命令
python ./src/mobile_use/main.py "Go to settings and tell me my current battery level"

# 数据抓取
python ./src/mobile_use/main.py \
  "Open Gmail, find all unread emails, and list their sender and subject line" \
  --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
```

## 六、SDK 使用

### 6.1 基本用法

```python
from minitap.mobile_use.sdk.agent import Agent
from minitap.mobile_use.sdk.builders.task_request_builder import TaskRequestBuilder

# 创建 Agent
agent = Agent()
await agent.init()

# 创建任务
task = TaskRequestBuilder() \
    .with_goal("Open Settings and check battery level") \
    .build()

# 执行任务
result = await agent.run(task)
print(result.output)
```

### 6.2 自定义配置

```python
from minitap.mobile_use.sdk.types.agent import AgentConfig

config = AgentConfig(
    device_id="emulator-5554",
    device_platform="android",
)

agent = Agent(config=config)
```

## 七、与其他方案对比

| 特性 | mobile-use | AppAgent | Droidrun |
|------|------------|----------|----------|
| **架构** | 多智能体 | 单智能体 | 多智能体 |
| **iOS 支持** | ✅ | ❌ | ❌ |
| **知识积累** | ❌ | ✅ | ❌ |
| **云设备支持** | ✅ (BrowserStack) | ❌ | ❌ |
| **数据抓取** | ✅ | ❌ | ❌ |
| **LLM 可配置** | ✅ | ✅ | ✅ |

## 八、优势与局限

### 8.1 核心优势

1. **跨平台支持**: 同时支持 Android 和 iOS
2. **多智能体架构**: 分工明确，可扩展性强
3. **数据抓取能力**: 可以从应用中提取结构化数据
4. **云设备集成**: 支持 BrowserStack 云测试平台
5. **灵活的 LLM 配置**: 每个智能体可以使用不同的模型

### 8.2 局限性

1. **游戏支持有限**: 游戏不提供 Accessibility Tree，效果受限
2. **iOS 物理设备**: 需要额外配置 WebDriverAgent
3. **依赖外部服务**: 需要 LLM API 密钥

## 九、参考资料

- [GitHub 仓库](https://github.com/minitap-ai/mobile-use)
- [官方文档](https://docs.minitap.ai/v2/mcp-server/introduction)
- [Discord 社区](https://discord.gg/6nSqmQ9pQs)
- [Benchmark 页面](https://minitap.ai/benchmark)
