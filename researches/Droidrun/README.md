# Droidrun 深度调研报告

> **作者**: Damon Li  
> **更新日期**: 2026年1月7日

## 一、项目概述

**Droidrun** 是一个基于 AI 大模型的 Android/iOS 自动化 Agent 框架，目前在 GitHub 上拥有 6.2K+ Star。它的核心理念是**将思考交给 AI，将执行交给框架**，从而打破了传统自动化脚本对特定 UI 控件的强依赖。

| 属性 | 信息 |
|------|------|
| **开发者** | Droidrun Team |
| **开源协议** | MIT License |
| **GitHub** | [droidrun/droidrun](https://github.com/droidrun/droidrun) |
| **文档** | [docs.droidrun.ai](https://docs.droidrun.ai) |
| **云服务** | [cloud.droidrun.ai](https://cloud.droidrun.ai) |
| **基准测试成绩** | 91.4% |

![Droidrun Logo](images/droidrun.png)

## 二、核心特性

### 2.1 多平台支持

- **Android**: 通过 ADB 控制真实设备或模拟器
- **iOS**: 支持 iOS 设备自动化

### 2.2 多 LLM 提供商支持

Droidrun 支持多种 LLM 提供商，用户可以根据需求选择：

| 提供商 | 支持状态 |
|--------|----------|
| OpenAI | ✅ |
| Anthropic | ✅ |
| Google Gemini | ✅ |
| Ollama (本地) | ✅ |
| DeepSeek | ✅ |

### 2.3 核心功能

1. **自然语言控制**: 使用自然语言命令控制设备
2. **规划能力**: 支持复杂多步任务的规划和执行
3. **截图分析**: 通过视觉理解设备当前状态
4. **执行追踪**: 使用 Arize Phoenix 进行执行追踪
5. **CLI 工具**: 提供易用的命令行界面
6. **Python API**: 可扩展的 Python API 用于自定义自动化

## 三、技术架构

### 3.1 多智能体协作架构

Droidrun 采用 **Manager-Executor** 多智能体协作架构：

```
┌─────────────────────────────────────────────────────────────┐
│                       DroidAgent                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Manager Agent                      │   │
│  │  - 分析当前状态                                       │   │
│  │  - 创建计划和子目标                                   │   │
│  │  - 跟踪进度                                          │   │
│  │  - 决定任务是否完成                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Executor Agent                      │   │
│  │  - 接收 Manager 的子目标                              │   │
│  │  - 分析当前 UI 状态                                   │   │
│  │  - 选择并执行适当的动作                               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 两种推理模式

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| `reasoning=False` | 直接使用 CodeActAgent 执行 | 简单任务 |
| `reasoning=True` | 使用 Manager + Executor 工作流 | 复杂多步任务 |

### 3.3 核心组件

1. **DroidAgent**: 顶层协调器，管理整个工作流
2. **ManagerAgent**: 规划和推理智能体
3. **ExecutorAgent**: 动作执行智能体
4. **CodeActAgent**: 直接代码执行智能体
5. **ScripterAgent**: 脚本生成智能体
6. **Tools**: 设备控制工具集

## 四、动作空间

Droidrun 定义了丰富的原子动作：

| 动作 | 参数 | 描述 |
|------|------|------|
| `click` | `index` | 点击指定索引的元素 |
| `long_press` | `index` | 长按指定索引的元素 |
| `type` | `text`, `index`, `clear` | 在指定元素中输入文本 |
| `swipe` | `direction` | 向指定方向滑动 |
| `system_button` | `button` | 按系统按钮（返回、主页等） |
| `open_app` | `app_name` | 打开指定应用 |
| `wait` | `seconds` | 等待指定秒数 |

## 五、App Card 机制

Droidrun 引入了 **App Card** 机制，为每个应用提供结构化的使用指南：

### 5.1 App Card 提供者模式

| 模式 | 描述 |
|------|------|
| `local` | 从本地目录加载 App Card |
| `server` | 从远程服务器加载 App Card |
| `composite` | 优先服务器，回退本地 |

### 5.2 App Card 内容

App Card 包含应用的使用说明、常见操作路径等信息，帮助 Agent 更好地理解和操作应用。

## 六、凭证管理

Droidrun 支持安全的凭证管理：

```python
# 支持多种凭证来源
credentials = {
    "SECRET_ID": "value"
}

# 或使用 CredentialManager
from droidrun.credential_manager import FileCredentialManager
cm = FileCredentialManager(credentials)
```

## 七、安装与使用

### 7.1 安装

```bash
pip install 'droidrun[google,anthropic,openai,deepseek,ollama,dev]'
```

### 7.2 快速开始

```python
from droidrun.agent.droid import DroidAgent

# 创建 Agent
agent = DroidAgent(
    goal="打开微信并发送消息给张三",
    config=config,
    llms=llm
)

# 运行任务
result = await agent.run()
```

### 7.3 CLI 使用

```bash
# 运行任务
droidrun run "打开设置并开启蓝牙"
```

## 八、性能评测

根据官方基准测试，Droidrun 在标准任务集上达到了 **91.4%** 的成功率。

## 九、优势与劣势

### 优势

1. **多 LLM 支持**: 支持 OpenAI、Anthropic、Gemini、Ollama、DeepSeek 等多种 LLM
2. **双平台支持**: 同时支持 Android 和 iOS
3. **模块化架构**: Manager-Executor 架构清晰，易于扩展
4. **App Card 机制**: 提供应用级别的使用指南
5. **MIT 协议**: 商业友好的开源协议
6. **云服务支持**: 提供云端托管服务

### 劣势

1. **无自有模型**: 依赖第三方 LLM，无专门训练的 GUI 模型
2. **框架复杂度**: 多智能体架构增加了系统复杂度
3. **延迟较高**: 多轮 LLM 调用可能导致较高延迟

## 十、与其他方案对比

| 特性 | Droidrun | UI-TARS | MAI-UI |
|------|----------|---------|--------|
| 自有模型 | ❌ | ✅ | ✅ |
| 多 LLM 支持 | ✅ | ❌ | ❌ |
| iOS 支持 | ✅ | ❌ | ❌ |
| 开源协议 | MIT | Apache 2.0 | Apache 2.0 |
| 云服务 | ✅ | ❌ | ❌ |

## 参考资料

1. [Droidrun GitHub Repository](https://github.com/droidrun/droidrun)
2. [Droidrun Documentation](https://docs.droidrun.ai)
3. [Droidrun Benchmark](https://droidrun.ai/benchmark)
