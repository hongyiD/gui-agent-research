# Droidrun 深度代码解构

> **作者**: Damon Li  
> **更新日期**: 2026年1月7日

## 一、项目结构

```
droidrun/
├── droidrun/                    # 核心代码目录
│   ├── __init__.py
│   ├── __main__.py              # CLI 入口
│   ├── agent/                   # 智能体模块
│   │   ├── codeact/             # CodeAct 智能体
│   │   │   ├── codeact_agent.py # 代码执行智能体
│   │   │   └── events.py        # 事件定义
│   │   ├── common/              # 公共组件
│   │   │   ├── constants.py     # 常量定义
│   │   │   └── events.py        # 公共事件
│   │   ├── droid/               # 主智能体
│   │   │   ├── droid_agent.py   # DroidAgent 核心实现
│   │   │   ├── events.py        # 事件定义
│   │   │   └── state.py         # 状态管理
│   │   ├── executor/            # 执行器智能体
│   │   │   ├── executor_agent.py# 动作执行
│   │   │   ├── events.py        # 事件定义
│   │   │   └── prompts.py       # 提示词
│   │   ├── manager/             # 管理器智能体
│   │   │   ├── manager_agent.py # 规划推理
│   │   │   ├── events.py        # 事件定义
│   │   │   ├── prompts.py       # 提示词
│   │   │   └── stateless_manager_agent.py
│   │   ├── oneflows/            # 单流程智能体
│   │   │   ├── app_starter_workflow.py
│   │   │   ├── structured_output_agent.py
│   │   │   └── text_manipulator.py
│   │   ├── scripter/            # 脚本生成智能体
│   │   │   ├── scripter_agent.py
│   │   │   ├── events.py
│   │   ├── trajectory/          # 轨迹记录
│   │   │   └── writer.py
│   │   ├── usage.py             # 使用统计
│   │   └── utils/               # 工具函数
│   │       ├── chat_utils.py    # 聊天工具
│   │       ├── executer.py      # 执行器
│   │       ├── inference.py     # 推理工具
│   │       ├── llm_loader.py    # LLM 加载
│   │       ├── llm_picker.py    # LLM 选择
│   │       ├── prompt_resolver.py# 提示词解析
│   │       ├── tools.py         # 工具定义
│   │       ├── tracing_setup.py # 追踪设置
│   │       └── trajectory.py    # 轨迹工具
│   ├── app_cards/               # App Card 模块
│   │   ├── app_card_provider.py # 提供者基类
│   │   └── providers/           # 具体提供者
│   │       ├── composite_provider.py
│   │       ├── local_provider.py
│   │       └── server_provider.py
│   ├── cli/                     # 命令行接口
│   │   └── logs.py
│   ├── config_manager/          # 配置管理
│   │   ├── config_manager.py    # 配置管理器
│   │   └── prompt_loader.py     # 提示词加载
│   ├── credential_manager/      # 凭证管理
│   ├── telemetry/               # 遥测模块
│   └── tools/                   # 设备控制工具
├── docs/                        # 文档目录
├── static/                      # 静态资源
├── pyproject.toml               # 项目配置
└── README.md                    # 项目说明
```

## 二、核心模块解析

### 2.1 DroidAgent (`droid_agent.py`)

DroidAgent 是整个框架的核心协调器，负责管理 Manager 和 Executor 之间的协作。

#### 类定义

```python
class DroidAgent(Workflow):
    """
    A wrapper class that coordinates between agents to achieve a user's goal.

    Reasoning modes:
    - reasoning=False: Uses CodeActAgent directly for immediate execution
    - reasoning=True: Uses ManagerAgent (planning) + ExecutorAgent (actions)
    """
```

#### 初始化参数

| 参数 | 类型 | 描述 |
|------|------|------|
| `goal` | `str` | 用户的目标或命令 |
| `config` | `DroidrunConfig` | 完整配置 |
| `llms` | `dict[str, LLM] \| LLM` | LLM 实例或字典 |
| `tools` | `Tools` | 设备控制工具实例 |
| `custom_tools` | `dict` | 自定义工具定义 |
| `credentials` | `dict \| CredentialManager` | 凭证管理 |
| `variables` | `dict` | 自定义变量 |
| `output_model` | `Type[BaseModel]` | 输出结构化模型 |
| `prompts` | `dict[str, str]` | 自定义提示词模板 |
| `timeout` | `int` | 工作流超时时间 |

#### LLM 分配

DroidAgent 支持为不同智能体分配不同的 LLM：

```python
if isinstance(llms, dict):
    self.manager_llm = llms.get("manager")
    self.executor_llm = llms.get("executor")
    self.codeact_llm = llms.get("codeact")
    self.text_manipulator_llm = llms.get("text_manipulator")
    self.app_opener_llm = llms.get("app_opener")
    self.scripter_llm = llms.get("scripter", self.codeact_llm)
    self.structured_output_llm = llms.get("structured_output", self.codeact_llm)
```

### 2.2 ManagerAgent (`manager_agent.py`)

ManagerAgent 负责任务规划和推理，是"大脑"角色。

#### 核心职责

1. **分析当前状态**: 理解设备当前的 UI 状态
2. **创建计划**: 将复杂任务分解为子目标
3. **跟踪进度**: 记录已完成的步骤
4. **决策**: 判断任务是否完成或需要调整

#### 系统提示词构建

```python
async def _build_system_prompt(self, has_text_to_modify: bool) -> str:
    variables = {
        "instruction": self.shared_state.instruction,
        "device_date": await self.tools_instance.get_date(),
        "app_card": self.shared_state.app_card,
        "error_history": error_history,
        "text_manipulation_enabled": has_text_to_modify,
        "custom_tools_descriptions": build_custom_tool_descriptions(self.custom_tools),
        "scripter_execution_enabled": self.agent_config.scripter.enabled,
        "available_secrets": available_secrets,
        "variables": self.shared_state.custom_variables,
        "output_schema": output_schema,
    }
    return await PromptLoader.load_prompt(prompt_path, variables)
```

#### App Card 提供者初始化

```python
def _initialize_app_card_provider(self) -> AppCardProvider:
    mode = self.app_card_config.mode.lower()
    
    if mode == "local":
        return LocalAppCardProvider(app_cards_dir=self.app_card_config.app_cards_dir)
    elif mode == "server":
        return ServerAppCardProvider(server_url=self.app_card_config.server_url)
    elif mode == "composite":
        return CompositeAppCardProvider(
            server_url=self.app_card_config.server_url,
            app_cards_dir=self.app_card_config.app_cards_dir
        )
```

### 2.3 ExecutorAgent (`executor_agent.py`)

ExecutorAgent 负责执行具体的动作，是"手脚"角色。

#### 核心职责

1. **接收子目标**: 从 Manager 获取当前要执行的子目标
2. **分析 UI**: 理解当前界面状态
3. **选择动作**: 决定执行哪个原子动作
4. **执行动作**: 调用设备控制 API

#### 动作执行逻辑

```python
async def _execute_action(self, action_dict: dict, description: str) -> tuple[bool, str, str]:
    action_type = action_dict.get("action", "unknown")

    # 优先检查自定义工具
    if action_type in self.custom_tools:
        return await self._execute_custom_tool(action_type, action_dict)

    # 执行内置动作
    if action_type == "click":
        index = action_dict.get("index")
        await click(index, tools=self.tools_instance)
        return True, "", f"Clicked element at index {index}"

    elif action_type == "type":
        text = action_dict.get("text")
        index = action_dict.get("index", -1)
        clear = action_dict.get("clear", False)
        await type(text, index, clear=clear, tools=self.tools_instance)
        return True, "", f"Typed '{text}' into element at index {index}"
    
    # ... 其他动作类型
```

### 2.4 工具定义 (`utils/tools.py`)

定义了所有可用的原子动作：

```python
ATOMIC_ACTION_SIGNATURES = {
    "click": {
        "description": "Click on an element by index",
        "parameters": {"index": "int - Element index to click"}
    },
    "long_press": {
        "description": "Long press on an element by index",
        "parameters": {"index": "int - Element index to long press"}
    },
    "type": {
        "description": "Type text into an element",
        "parameters": {
            "text": "str - Text to type",
            "index": "int - Element index (optional)",
            "clear": "bool - Clear existing text first"
        }
    },
    "swipe": {
        "description": "Swipe in a direction",
        "parameters": {"direction": "str - up/down/left/right"}
    },
    "system_button": {
        "description": "Press a system button",
        "parameters": {"button": "str - back/home/recent"}
    },
    "open_app": {
        "description": "Open an application",
        "parameters": {"app_name": "str - Name of the app"}
    },
    "wait": {
        "description": "Wait for a specified time",
        "parameters": {"seconds": "float - Seconds to wait"}
    }
}
```

## 三、状态管理 (`state.py`)

### DroidAgentState

统一的状态管理类，在所有智能体之间共享：

```python
class DroidAgentState:
    instruction: str           # 用户指令
    plan: str                  # 当前计划
    progress_summary: str      # 进度摘要
    memory: str                # 记忆内容
    
    # 历史记录
    action_history: list       # 动作历史
    summary_history: list      # 摘要历史
    error_descriptions: list   # 错误描述
    action_outcomes: list      # 动作结果
    message_history: list      # 消息历史
    
    # 当前状态
    screenshot: bytes          # 当前截图
    formatted_device_state: str# 格式化的设备状态
    app_card: str              # 当前应用的 App Card
    
    # 上一步信息
    last_thought: str          # 上一步思考
    last_action: dict          # 上一步动作
    last_summary: str          # 上一步摘要
    
    # 错误处理
    error_flag_plan: bool      # 计划错误标志
    err_to_manager_thresh: int # 错误阈值
    
    # 自定义变量
    custom_variables: dict     # 用户自定义变量
```

## 四、事件系统

Droidrun 使用基于 LlamaIndex Workflow 的事件驱动架构：

### 4.1 Manager 事件

```python
class ManagerInputEvent(Event):
    """Manager 输入事件"""
    pass

class ManagerPlanEvent(Event):
    """Manager 计划事件"""
    plan: str
    subgoal: str
    progress: str
```

### 4.2 Executor 事件

```python
class ExecutorInputEvent(Event):
    """Executor 输入事件"""
    subgoal: str

class ExecutorResultEvent(Event):
    """Executor 结果事件"""
    action: dict
    success: bool
    error: str
    summary: str
```

## 五、配置系统

### 5.1 配置结构

```python
class DroidrunConfig(BaseModel):
    agent: AgentConfig           # 智能体配置
    device: DeviceConfig         # 设备配置
    tools: ToolsConfig           # 工具配置
    logging: LoggingConfig       # 日志配置
    tracing: TracingConfig       # 追踪配置
    telemetry: TelemetryConfig   # 遥测配置
    llm_profiles: dict           # LLM 配置文件
    credentials: CredentialsConfig# 凭证配置
```

### 5.2 智能体配置

```python
class AgentConfig(BaseModel):
    manager: ManagerConfig       # Manager 配置
    executor: ExecutorConfig     # Executor 配置
    scripter: ScripterConfig     # Scripter 配置
    app_cards: AppCardsConfig    # App Card 配置
    streaming: bool = True       # 是否流式输出
    after_sleep_action: float    # 动作后等待时间
```

## 六、LLM 加载机制

### 6.1 LLM 加载器 (`llm_loader.py`)

```python
def load_agent_llms(config: DroidrunConfig, output_model=None, **kwargs) -> dict[str, LLM]:
    """
    从配置加载各智能体的 LLM 实例
    """
    llms = {}
    
    # 加载各智能体的 LLM
    for agent_name in ["manager", "executor", "codeact", "text_manipulator", "app_opener", "scripter"]:
        profile_name = getattr(config.agent, f"{agent_name}_profile", None)
        if profile_name and profile_name in config.llm_profiles:
            llms[agent_name] = create_llm_from_profile(config.llm_profiles[profile_name])
    
    return llms
```

### 6.2 支持的 LLM 提供商

```python
# OpenAI
from llama_index.llms.openai import OpenAI

# Anthropic
from llama_index.llms.anthropic import Anthropic

# Google Gemini
from llama_index.llms.gemini import Gemini

# Ollama (本地)
from llama_index.llms.ollama import Ollama

# DeepSeek
from llama_index.llms.deepseek import DeepSeek
```

## 七、追踪与遥测

### 7.1 Arize Phoenix 集成

```python
def setup_tracing(config: TracingConfig, agent=None):
    """设置执行追踪"""
    if config.enabled:
        # 配置 Arize Phoenix
        from arize_phoenix import init_tracing
        init_tracing(
            project_name=config.project_name,
            endpoint=config.endpoint
        )
```

### 7.2 Langfuse 集成

```python
def record_langfuse_screenshot(screenshot: bytes, name: str):
    """记录截图到 Langfuse"""
    # 将截图上传到 Langfuse 用于可视化调试
    pass
```

## 八、工作流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           用户输入目标                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DroidAgent 初始化                            │
│  - 加载配置                                                          │
│  - 初始化 LLM                                                        │
│  - 创建 Tools 实例                                                   │
│  - 设置追踪                                                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         获取设备状态                                  │
│  - 截取屏幕截图                                                      │
│  - 解析 UI 元素树                                                    │
│  - 加载 App Card                                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ManagerAgent 规划                              │
│  - 分析当前状态                                                      │
│  - 生成计划和子目标                                                  │
│  - 更新进度                                                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ExecutorAgent 执行                              │
│  - 接收子目标                                                        │
│  - 选择动作                                                          │
│  - 执行动作                                                          │
│  - 返回结果                                                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────┐
                        │   任务完成？       │
                        └───────────────────┘
                           │           │
                          否           是
                           │           │
                           ▼           ▼
                    返回 Manager    返回结果
```

## 九、关键设计模式

### 9.1 工作流模式 (Workflow Pattern)

基于 LlamaIndex Workflow，使用 `@step` 装饰器定义工作流步骤：

```python
class ExecutorAgent(Workflow):
    @step
    async def prepare_context(self, ctx: Context, ev: StartEvent) -> ExecutorContextEvent:
        # 准备上下文
        pass
    
    @step
    async def get_response(self, ctx: Context, ev: ExecutorContextEvent) -> ExecutorResponseEvent:
        # 获取 LLM 响应
        pass
    
    @step
    async def execute(self, ctx: Context, ev: ExecutorActionEvent) -> ExecutorActionResultEvent:
        # 执行动作
        pass
```

### 9.2 提供者模式 (Provider Pattern)

App Card 使用提供者模式，支持多种数据源：

```python
class AppCardProvider(ABC):
    @abstractmethod
    async def load_app_card(self, package_name: str, instruction: str = "") -> str:
        pass

class LocalAppCardProvider(AppCardProvider):
    async def load_app_card(self, package_name: str, instruction: str = "") -> str:
        # 从本地文件加载
        pass

class ServerAppCardProvider(AppCardProvider):
    async def load_app_card(self, package_name: str, instruction: str = "") -> str:
        # 从服务器加载
        pass
```

### 9.3 凭证管理模式

```python
class CredentialManager(ABC):
    @abstractmethod
    async def get_keys(self) -> list[str]:
        pass
    
    @abstractmethod
    async def get_secret(self, key: str) -> str:
        pass

class FileCredentialManager(CredentialManager):
    def __init__(self, credentials: dict):
        self.secrets = credentials
```

## 十、总结

Droidrun 是一个设计精良的 GUI Agent 框架，其核心优势在于：

1. **模块化架构**: Manager-Executor 分离，职责清晰
2. **多 LLM 支持**: 灵活选择不同的 LLM 提供商
3. **事件驱动**: 基于 LlamaIndex Workflow 的事件系统
4. **可扩展性**: 支持自定义工具和提示词
5. **追踪能力**: 集成 Arize Phoenix 和 Langfuse

但也存在一些局限：

1. **无自有模型**: 依赖第三方 LLM，无专门训练的 GUI 模型
2. **延迟较高**: 多智能体协作增加了 LLM 调用次数
3. **复杂度**: 框架本身较为复杂，学习曲线陡峭
