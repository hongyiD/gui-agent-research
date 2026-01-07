# mobile-use 深度代码解构

> **作者**: Damon Li  
> **更新日期**: 2026年1月7日

## 一、项目结构

```
mobile-use/
├── minitap/
│   └── mobile_use/
│       ├── agents/                    # 智能体模块
│       │   ├── contextor/            # 上下文管理器
│       │   ├── cortex/               # 决策大脑
│       │   ├── executor/             # 执行器
│       │   ├── hopper/               # 应用跳转器
│       │   ├── orchestrator/         # 编排器
│       │   ├── outputter/            # 输出器
│       │   ├── planner/              # 任务规划器
│       │   └── summarizer/           # 总结器
│       ├── clients/                   # 设备客户端
│       │   ├── browserstack_client.py
│       │   ├── idb_client.py         # iOS fb-idb 客户端
│       │   ├── ios_client.py
│       │   ├── ui_automator_client.py # Android UIAutomator
│       │   └── wda_client.py         # iOS WebDriverAgent
│       ├── controllers/               # 设备控制器
│       │   ├── android_controller.py
│       │   ├── ios_controller.py
│       │   ├── unified_controller.py
│       │   └── controller_factory.py
│       ├── graph/                     # LangGraph 状态图
│       │   ├── graph.py
│       │   └── state.py
│       ├── sdk/                       # SDK 接口
│       │   ├── agent.py              # 主 Agent 类
│       │   ├── builders/             # 构建器
│       │   └── examples/             # 示例代码
│       ├── services/                  # 服务层
│       │   ├── llm.py                # LLM 服务
│       │   └── telemetry.py          # 遥测服务
│       ├── tools/                     # 工具定义
│       ├── utils/                     # 工具函数
│       ├── config.py                  # 配置管理
│       ├── constants.py               # 常量定义
│       ├── context.py                 # 上下文定义
│       └── main.py                    # CLI 入口
├── doc/                               # 文档和图片
├── scripts/                           # 辅助脚本
├── tests/                             # 测试代码
├── mobile-use.sh                      # Docker 启动脚本
├── llm-config.defaults.jsonc          # 默认 LLM 配置
└── pyproject.toml                     # 项目配置
```

## 二、核心模块解析

### 2.1 Agent 类 (`sdk/agent.py`)

这是 mobile-use 的主入口类，负责初始化和任务执行。

#### 初始化流程

```python
class Agent:
    def __init__(self, *, config: AgentConfig | None = None):
        self._config = config or get_default_agent_config()
        self._tasks = []
        self._tmp_traces_dir = Path(tempfile.gettempdir()) / "mobile-use-traces"
        self._initialized = False
        self._task_lock = asyncio.Lock()
        
        # 初始化平台服务（如果有 API 密钥）
        if settings.MINITAP_API_KEY:
            self._platform_service = PlatformService()
            self._cloud_mobile_service = CloudMobileService()

    async def init(self, api_key: str | None = None, ...):
        """初始化 Agent，连接设备"""
        # 1. 检测可用设备
        device_id, platform, ios_device_type = get_first_device(logger=logger)
        
        # 2. 初始化客户端
        self._init_clients(device_id, platform, ios_device_type, ...)
        
        # 3. 启动必要的服务器
        self._run_servers(device_id, platform)
        
        # 4. 获取设备上下文
        self._device_context = await self._get_device_context(device_id, platform)
        
        self._initialized = True
```

#### 任务执行

```python
async def run(self, task_request: TaskRequest, ...):
    """执行任务"""
    # 1. 创建执行上下文
    ctx = MobileUseContext(
        device=self._device_context,
        trace_id=trace_id,
        execution_setup=execution_setup,
        ...
    )
    
    # 2. 获取 LangGraph 图
    graph = get_graph(ctx)
    
    # 3. 初始化状态
    initial_state = State(
        initial_goal=task_request.goal,
        output_description=task_request.output_description,
        ...
    )
    
    # 4. 执行图
    async for event in graph.astream(initial_state, config=config):
        # 处理事件
        ...
    
    # 5. 返回结果
    return Task(
        id=trace_id,
        goal=task_request.goal,
        output=final_state.output,
        status=TaskStatus.COMPLETED,
    )
```

### 2.2 状态图 (`graph/graph.py`)

mobile-use 使用 LangGraph 构建状态机，管理智能体之间的协作。

#### 图结构

```python
def get_graph(ctx: MobileUseContext):
    """构建 LangGraph 状态图"""
    builder = StateGraph(State)
    
    # 添加节点
    builder.add_node("planner", PlannerNode(ctx))
    builder.add_node("orchestrator", OrchestratorNode(ctx))
    builder.add_node("contextor", ContextorNode(ctx))
    builder.add_node("cortex", CortexNode(ctx))
    builder.add_node("executor", ExecutorNode(ctx))
    builder.add_node("summarizer", SummarizerNode(ctx))
    builder.add_node("outputter", OutputterNode(ctx))
    builder.add_node("hopper", HopperNode(ctx))
    
    # 设置入口
    builder.set_entry_point("planner")
    
    # 添加边
    builder.add_edge("planner", "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        should_continue_or_end,
        {
            "continue": "contextor",
            "end": "summarizer",
            "replan": "planner",
        }
    )
    builder.add_edge("contextor", "cortex")
    builder.add_conditional_edges(
        "cortex",
        should_execute_or_hop,
        {
            "execute": "executor",
            "hop": "hopper",
        }
    )
    builder.add_edge("executor", "orchestrator")
    builder.add_edge("hopper", "orchestrator")
    builder.add_edge("summarizer", "outputter")
    builder.add_edge("outputter", END)
    
    return builder.compile()
```

#### 状态定义

```python
class State(TypedDict):
    # 任务信息
    initial_goal: str
    output_description: str | None
    
    # 规划信息
    subgoal_plan: list[Subgoal]
    complete_subgoals_by_ids: list[str]
    
    # 设备状态
    latest_screenshot: bytes | None
    latest_ui_hierarchy: list[dict] | None
    focused_app_info: str | None
    device_date: str | None
    
    # 智能体思考
    agents_thoughts: list[str]
    structured_decisions: str | None
    cortex_last_thought: str | None
    
    # 执行器消息
    executor_messages: list[BaseMessage]
    
    # 输出
    output: str | None
```

### 2.3 Planner 规划器 (`agents/planner/planner.py`)

负责将用户目标分解为子目标列表。

```python
class PlannerNode:
    def __init__(self, ctx: MobileUseContext):
        self.ctx = ctx
    
    async def __call__(self, state: State):
        # 1. 构建提示词
        system_message = Template(
            Path(__file__).parent.joinpath("planner.md").read_text()
        ).render(
            platform=self.ctx.device.mobile_platform.value,
            initial_goal=state.initial_goal,
        )
        
        # 2. 调用 LLM
        llm = get_llm(ctx=self.ctx, name="planner", temperature=1)
        llm = llm.with_structured_output(PlannerOutput)
        response: PlannerOutput = await llm.ainvoke(messages)
        
        # 3. 返回子目标计划
        return {
            "subgoal_plan": response.subgoals,
            "agents_thoughts": [f"Plan: {response.subgoals}"],
        }
```

### 2.4 Orchestrator 编排器 (`agents/orchestrator/orchestrator.py`)

管理子目标的执行状态，决定下一步行动。

```python
class OrchestratorNode:
    async def __call__(self, state: State):
        no_subgoal_started = nothing_started(state.subgoal_plan)
        current_subgoal = get_current_subgoal(state.subgoal_plan)
        
        # 如果没有开始任何子目标，启动第一个
        if no_subgoal_started or not current_subgoal:
            state.subgoal_plan = start_next_subgoal(state.subgoal_plan)
            return {"subgoal_plan": state.subgoal_plan, ...}
        
        # 检查需要评估的子目标
        subgoals_to_examine = get_subgoals_by_ids(
            subgoals=state.subgoal_plan,
            ids=state.complete_subgoals_by_ids,
        )
        
        # 调用 LLM 评估
        llm = get_llm(ctx=self.ctx, name="orchestrator", temperature=1)
        llm = llm.with_structured_output(OrchestratorOutput)
        response: OrchestratorOutput = await llm.ainvoke(messages)
        
        # 根据评估结果更新状态
        if response.needs_replaning:
            state.subgoal_plan = fail_current_subgoal(state.subgoal_plan)
            return {"subgoal_plan": state.subgoal_plan, ...}
        
        state.subgoal_plan = complete_subgoals_by_ids(
            subgoals=state.subgoal_plan,
            ids=response.completed_subgoal_ids,
        )
        
        if all_completed(state.subgoal_plan):
            return {"subgoal_plan": state.subgoal_plan, ...}
        
        state.subgoal_plan = start_next_subgoal(state.subgoal_plan)
        return {"subgoal_plan": state.subgoal_plan, ...}
```

### 2.5 Cortex 决策大脑 (`agents/cortex/cortex.py`)

分析屏幕状态，做出操作决策。

```python
class CortexNode:
    async def __call__(self, state: State):
        # 1. 获取执行器反馈
        executor_feedback = get_executor_agent_feedback(state)
        
        # 2. 构建提示词
        system_message = Template(
            Path(__file__).parent.joinpath("cortex.md").read_text()
        ).render(
            platform=self.ctx.device.mobile_platform.value,
            initial_goal=state.initial_goal,
            subgoal_plan=state.subgoal_plan,
            current_subgoal=get_current_subgoal(state.subgoal_plan),
            executor_feedback=executor_feedback,
            executor_tools_list=format_tools_list(...),
        )
        
        # 3. 添加 UI 层次结构
        if state.latest_ui_hierarchy:
            ui_hierarchy_str = json.dumps(state.latest_ui_hierarchy, indent=2)
            messages.append(HumanMessage(content="UI hierarchy:\n" + ui_hierarchy_str))
        
        # 4. 添加屏幕截图
        if state.latest_screenshot:
            compressed_image_base64 = controller.get_compressed_b64_screenshot(
                state.latest_screenshot
            )
            messages.append(get_screenshot_message_for_llm(compressed_image_base64))
        
        # 5. 调用 LLM
        llm = get_llm(ctx=self.ctx, name="cortex", temperature=1)
        llm = llm.with_structured_output(CortexOutput)
        response: CortexOutput = await llm.ainvoke(messages)
        
        # 6. 返回决策
        return {
            "structured_decisions": response.decisions,
            "complete_subgoals_by_ids": response.complete_subgoals_by_ids,
            ...
        }
```

### 2.6 Executor 执行器 (`agents/executor/executor.py`)

将决策转换为具体的设备操作。

```python
class ExecutorNode:
    async def __call__(self, state: State):
        # 1. 解析决策
        decisions = state.structured_decisions
        
        # 2. 创建设备控制器
        controller = create_device_controller(self.ctx)
        
        # 3. 执行操作
        for decision in decisions:
            if decision.action == "tap":
                await controller.tap(decision.x, decision.y)
            elif decision.action == "type_text":
                await controller.type_text(decision.text)
            elif decision.action == "swipe":
                await controller.swipe(
                    decision.start_x, decision.start_y,
                    decision.end_x, decision.end_y
                )
            # ... 其他操作
        
        # 4. 返回执行结果
        return {
            "executor_messages": [ToolMessage(content=result, ...)],
            ...
        }
```

### 2.7 设备控制器 (`controllers/`)

#### Android 控制器

```python
class AndroidController(DeviceController):
    def __init__(self, ctx: MobileUseContext):
        self.ctx = ctx
        self.adb_client = ctx.adb_client
        self.ui_client = ctx.ui_adb_client
    
    async def tap(self, x: int, y: int):
        """点击指定坐标"""
        await self.ui_client.click(x, y)
    
    async def type_text(self, text: str):
        """输入文本"""
        await self.ui_client.set_text(text)
    
    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300):
        """滑动"""
        await self.ui_client.swipe(x1, y1, x2, y2, duration)
    
    async def get_screenshot(self) -> bytes:
        """获取屏幕截图"""
        return await self.ui_client.screenshot()
    
    async def get_ui_hierarchy(self) -> list[dict]:
        """获取 UI 层次结构"""
        xml = await self.ui_client.dump_hierarchy()
        return parse_xml_to_dict(xml)
    
    async def press_key(self, key: str):
        """按下按键"""
        key_code = KEY_CODES.get(key)
        await self.adb_client.shell(f"input keyevent {key_code}")
    
    async def open_app(self, package: str):
        """打开应用"""
        await self.adb_client.shell(
            f"monkey -p {package} -c android.intent.category.LAUNCHER 1"
        )
```

#### iOS 控制器

```python
class IosController(DeviceController):
    def __init__(self, ctx: MobileUseContext):
        self.ctx = ctx
        self.ios_client = ctx.ios_client
    
    async def tap(self, x: int, y: int):
        """点击指定坐标"""
        await self.ios_client.tap(x, y)
    
    async def type_text(self, text: str):
        """输入文本"""
        await self.ios_client.text_input(text)
    
    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300):
        """滑动"""
        await self.ios_client.swipe(x1, y1, x2, y2, duration / 1000)
    
    async def get_screenshot(self) -> bytes:
        """获取屏幕截图"""
        return await self.ios_client.screenshot()
    
    async def get_ui_hierarchy(self) -> list[dict]:
        """获取 UI 层次结构"""
        return await self.ios_client.accessibility_info()
```

## 三、LLM 服务 (`services/llm.py`)

### 3.1 模型配置

```python
def get_llm(ctx: MobileUseContext, name: str, temperature: float = 0.7, use_fallback: bool = False):
    """获取指定智能体的 LLM 实例"""
    # 1. 加载配置
    config = load_llm_config()
    agent_config = config.get(name, config.get("default"))
    
    # 2. 选择提供商
    provider = agent_config.get("provider", "openai")
    model = agent_config.get("model", "gpt-4o")
    
    # 3. 创建 LLM 实例
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )
    elif provider == "google_vertex":
        return ChatVertexAI(
            model=model,
            temperature=temperature,
        )
    # ... 其他提供商
```

### 3.2 Fallback 机制

```python
async def with_fallback(main_call, fallback_call):
    """带 fallback 的 LLM 调用"""
    try:
        return await main_call()
    except Exception as e:
        logger.warning(f"Main LLM call failed: {e}, trying fallback...")
        return await fallback_call()
```

## 四、数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           mobile-use 数据流                                  │
└─────────────────────────────────────────────────────────────────────────────┘

用户输入: "Open Gmail and list unread emails"
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Planner                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 输入: initial_goal                                                      │ │
│  │ 输出: subgoal_plan = [                                                  │ │
│  │   {id: 1, goal: "Open Gmail app"},                                      │ │
│  │   {id: 2, goal: "Navigate to inbox"},                                   │ │
│  │   {id: 3, goal: "Find unread emails"},                                  │ │
│  │   {id: 4, goal: "Extract sender and subject"}                           │ │
│  │ ]                                                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Orchestrator                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 输入: subgoal_plan, complete_subgoals_by_ids                            │ │
│  │ 输出: current_subgoal = {id: 1, goal: "Open Gmail app", status: "in_progress"} │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Contextor                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 获取设备状态:                                                           │ │
│  │ - latest_screenshot: <bytes>                                            │ │
│  │ - latest_ui_hierarchy: [{...}, {...}]                                   │ │
│  │ - focused_app_info: "com.android.launcher3"                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               Cortex                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 输入: screenshot, ui_hierarchy, current_subgoal                         │ │
│  │ 分析: "I see the home screen. Gmail icon is at position (540, 1200)"    │ │
│  │ 输出: structured_decisions = "tap(540, 1200)"                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Executor                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 输入: structured_decisions = "tap(540, 1200)"                           │ │
│  │ 执行: controller.tap(540, 1200)                                         │ │
│  │ 输出: executor_messages = [ToolMessage("Tapped at (540, 1200)")]        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
        循环回到 Orchestrator，直到所有子目标完成
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Summarizer                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 汇总执行结果                                                            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Outputter                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 格式化输出:                                                             │ │
│  │ [                                                                       │ │
│  │   {"sender": "john@example.com", "subject": "Meeting tomorrow"},        │ │
│  │   {"sender": "jane@example.com", "subject": "Project update"}           │ │
│  │ ]                                                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 五、配置系统

### 5.1 环境变量

```bash
# .env
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，用于自定义端点
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json  # Vertex AI
MINITAP_API_KEY=xxx  # Minitap 平台 API
```

### 5.2 LLM 配置

```jsonc
// llm-config.defaults.jsonc
{
  "default": {
    "provider": "openai",
    "model": "gpt-4o"
  },
  "planner": {
    "provider": "openai",
    "model": "gpt-4o"
  },
  "cortex": {
    "provider": "openai",
    "model": "gpt-4o"
  },
  "orchestrator": {
    "provider": "openai",
    "model": "gpt-4o-mini"
  }
}
```

## 六、扩展性

### 6.1 添加新智能体

1. 在 `agents/` 目录下创建新模块
2. 实现 `__call__` 方法，接收 `State`，返回更新
3. 在 `graph/graph.py` 中添加节点和边

```python
# agents/my_agent/my_agent.py
class MyAgentNode:
    def __init__(self, ctx: MobileUseContext):
        self.ctx = ctx
    
    async def __call__(self, state: State):
        # 处理逻辑
        return {"agents_thoughts": ["My agent processed"], ...}
```

### 6.2 添加新设备支持

1. 在 `clients/` 目录下创建新客户端
2. 在 `controllers/` 目录下创建新控制器
3. 在 `controller_factory.py` 中注册

```python
# controllers/my_device_controller.py
class MyDeviceController(DeviceController):
    async def tap(self, x: int, y: int):
        # 实现点击
        pass
    
    async def get_screenshot(self) -> bytes:
        # 实现截图
        pass
```

## 七、总结

mobile-use 的代码架构具有以下特点：

1. **模块化设计**: 智能体、控制器、服务层分离清晰
2. **LangGraph 状态机**: 使用 LangGraph 管理复杂的智能体协作
3. **多平台支持**: 统一的控制器接口，支持 Android 和 iOS
4. **灵活的 LLM 配置**: 每个智能体可以使用不同的模型
5. **可扩展性强**: 易于添加新智能体和新设备支持

这种架构使得 mobile-use 能够灵活应对各种移动自动化场景，同时保持代码的可维护性和可扩展性。
