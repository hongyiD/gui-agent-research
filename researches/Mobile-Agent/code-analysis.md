# Mobile-Agent 深度代码解构

> **作者**: Damon Li  
> **更新日期**: 2026年1月6日

本报告对 [X-PLUG/MobileAgent](https://github.com/X-PLUG/MobileAgent) 项目进行深度代码解构，重点分析其核心组件 **Mobile-Agent-v3** 和 **GUI-Owl** 的实现细节。

## 1. 项目结构分析

Mobile-Agent 仓库是一个项目集合，包含了从 v1 到 v3 的多个版本以及相关的子项目。v3 版本是当前最核心和最先进的实现。

```bash
MobileAgent/
├── Mobile-Agent-v3/                # v3 核心代码
│   ├── README.md
│   ├── android_world_v3/         # 针对 AndroidWorld 基准测试的代码
│   │   └── android_world/
│   │       └── agents/
│   │           ├── gui_owl.py        # GUI-Owl 模型的核心实现
│   │           ├── mobile_agent_v3.py  # Mobile-Agent-v3 多智能体框架实现
│   │           ├── mobile_agent_v3_agent.py  # 多智能体定义
│   │           └── new_json_action.py  # 动作空间定义
│   ├── mobile_v3/                  # 在真实设备上运行 v3 的代码
│   │   ├── run_mobileagentv3.py    # 主运行脚本
│   │   └── utils/
│   │       ├── mobile_agent_e.py   # 多智能体框架定义 (Manager, Executor等)
│   │       ├── controller.py       # 设备控制器基类
│   │       ├── android_controller.py # Android ADB 控制器
│   │       └── harmonyos_controller.py # HarmonyOS HDC 控制器
│   └── os_world_v3/                # 针对 OSWorld 基准测试的代码
├── Mobile-Agent-v2/                # v2 版本代码
├── Mobile-Agent-v1/                # v1 版本代码
└── README.md                       # 主项目 README
```

## 2. 核心架构：多智能体协作框架

Mobile-Agent v3 采用了创新的多智能体协作架构，包含四个专门化的智能体：

### 2.1 InfoPool（信息池）

`InfoPool` 是整个系统的核心数据结构，负责在多个智能体之间共享状态和信息。

```python
@dataclass
class InfoPool:
    """跨智能体信息共享池"""
    
    # 用户输入和知识
    instruction: str = ""                    # 用户指令
    additional_knowledge_manager: str = ""   # Manager 额外知识
    additional_knowledge_executor: str = ""  # Executor 额外知识
    
    # 工作记忆
    summary_history: list = field(default_factory=list)   # 动作描述历史
    action_history: list = field(default_factory=list)    # 动作历史
    action_outcomes: list = field(default_factory=list)   # 动作结果 (A/B/C)
    error_descriptions: list = field(default_factory=list) # 错误描述
    
    # 规划状态
    plan: str = ""                    # 当前计划
    completed_plan: str = ""          # 已完成的子目标
    important_notes: str = ""         # 重要笔记
    
    # 错误处理
    error_flag_plan: bool = False     # 是否需要重新规划
    err_to_manager_thresh: int = 2    # 连续错误阈值
```

### 2.2 Manager（管理者）

Manager 负责高层规划，将复杂任务分解为子目标，并根据执行情况动态调整计划。

**核心职责**：
- 首次规划：根据用户指令制定高层计划
- 进度跟踪：记录已完成的子目标
- 错误升级：当 Executor 连续失败时，重新规划

**关键代码逻辑**：

```python
class Manager(BaseAgent):
    def get_prompt(self, info_pool: InfoPool) -> str:
        if info_pool.plan == "":
            # 首次规划
            prompt += "Make a high-level plan to achieve the user's request..."
        else:
            # 进度检查和计划更新
            prompt += f"### Historical Operations ###\n{info_pool.completed_plan}"
            prompt += f"### Plan ###\n{info_pool.plan}"
            
            if info_pool.error_flag_plan:
                # 错误升级：提供最近失败的动作日志
                prompt += "### Potentially Stuck! ###\n"
                for act, summ, err_des in zip(recent_actions, recent_summaries, recent_err_des):
                    prompt += f"- Attempt: Action: {act} | Outcome: Failed | Feedback: {err_des}\n"
```

**输出格式**：
```
### Thought ###
规划理由和分析

### Historical Operations ###
已完成的子目标列表

### Plan ###
1. 第一个子目标
2. 第二个子目标
...
```

### 2.3 Executor（执行者）

Executor 负责根据当前计划和屏幕状态，选择并执行具体的原子动作。

**支持的原子动作**：

| 动作类型 | 参数 | 描述 |
|---------|------|------|
| `click` | `coordinate: [x, y]` | 点击指定坐标 |
| `long_press` | `coordinate: [x, y]` | 长按指定坐标 |
| `type` | `text: string` | 在激活的输入框中输入文本 |
| `swipe` | `coordinate, coordinate2` | 从起点滑动到终点 |
| `open_app` | `text: app_name` | 打开指定应用 |
| `system_button` | `button: Home/Back/Enter` | 按系统按钮 |
| `answer` | `text: string` | 回答用户问题 |

**关键代码逻辑**：

```python
class Executor(BaseAgent):
    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt += f"### Overall Plan ###\n{info_pool.plan}"
        prompt += f"### Current Subgoal ###\n{truncated_current_goal}"
        
        # 最近动作历史（最多5条）
        prompt += "### Latest Action History ###\n"
        for act, summ, outcome, err_des in zip(...):
            if outcome == "A":
                prompt += f"Action: {act} | Outcome: Successful\n"
            else:
                prompt += f"Action: {act} | Outcome: Failed | Feedback: {err_des}\n"
```

### 2.4 ActionReflector（动作反思者）

ActionReflector 负责验证上一个动作是否成功执行，并提供错误反馈。

**输出分类**：
- **A**: 成功或部分成功
- **B**: 失败，进入了错误页面，需要返回
- **C**: 失败，动作没有产生任何变化

```python
class ActionReflector(BaseAgent):
    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt += "### Latest Action ###\n"
        prompt += f"Action: {info_pool.last_action}\n"
        prompt += f"Expectation: {info_pool.last_summary}\n"
        
        prompt += "### Outcome ###\n"
        prompt += "A: Successful or Partially Successful\n"
        prompt += "B: Failed. Wrong page, need to return\n"
        prompt += "C: Failed. No changes produced\n"
```

### 2.5 Notetaker（笔记员）

Notetaker 负责记录任务执行过程中的重要信息，如需要记住的数字、名称等。

```python
class Notetaker(BaseAgent):
    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt += f"### Existing Important Notes ###\n{info_pool.important_notes}"
        prompt += "Carefully examine the information above to identify any important content..."
```

## 3. 核心工作流解析 (`run_mobileagentv3.py`)

`mobile_v3/run_mobileagentv3.py` 是在真实设备上启动 Mobile-Agent-v3 的入口脚本。

```python
def run_instruction(adb_path, hdc_path, api_key, base_url, model, instruction, ...):
    # 1. 初始化设备控制器 (Android/HarmonyOS)
    if adb_path:
        controller = AndroidController(adb_path)
    else:
        controller = HarmonyOSController(hdc_path)

    # 2. 初始化信息池和四个智能体
    info_pool = InfoPool(...)
    vllm = GUIOwlWrapper(api_key, base_url, model)
    manager = Manager()
    executor = Executor()
    notetaker = Notetaker()
    action_reflector = ActionReflector()

    # 3. 主循环，最多执行 max_step 次
    for step in range(max_step):
        # 3.1 获取当前屏幕截图
        controller.get_screenshot(local_image_dir)
        
        # 3.2 错误升级检查
        if len(info_pool.action_outcomes) >= err_to_manager_thresh:
            latest_outcomes = info_pool.action_outcomes[-err_to_manager_thresh:]
            if all(outcome in ["B", "C"] for outcome in latest_outcomes):
                info_pool.error_flag_plan = True

        # 3.3 调用 Manager 进行规划
        if not skip_manager:
            prompt_planning = manager.get_prompt(info_pool)
            output_planning = vllm.predict_mm(prompt_planning, [screenshot])
            info_pool.plan = manager.parse_response(output_planning)['plan']

        # 如果计划完成，则退出
        if "Finished" in info_pool.plan:
            break

        # 3.4 调用 Executor 生成具体动作
        prompt_action = executor.get_prompt(info_pool)
        output_action = vllm.predict_mm(prompt_action, [screenshot])
        action = executor.parse_response(output_action)['action']

        # 3.5 执行动作
        execute_action(controller, json.loads(action))

        # 3.6 获取动作后截图
        controller.get_screenshot(after_screenshot)

        # 3.7 调用 ActionReflector 进行反思
        prompt_reflect = action_reflector.get_prompt(info_pool)
        output_reflect = vllm.predict_mm(prompt_reflect, [before, after])
        outcome = action_reflector.parse_response(output_reflect)['outcome']
        info_pool.action_outcomes.append(outcome)

        # 3.8 如果成功且启用 Notetaker
        if outcome == "A" and if_notetaker:
            prompt_note = notetaker.get_prompt(info_pool)
            output_note = vllm.predict_mm(prompt_note, [after_screenshot])
            info_pool.important_notes = notetaker.parse_response(output_note)['important_notes']
```

## 4. 错误处理机制

Mobile-Agent v3 实现了智能的错误处理机制：

### 4.1 动作级错误处理

ActionReflector 检测每个动作的执行结果，分为三种状态：
- **A (成功)**: 动作达到预期效果
- **B (错误页面)**: 进入了错误的页面，需要返回
- **C (无变化)**: 动作没有产生任何变化

### 4.2 错误升级机制

当连续 2 次失败后，将错误信息升级给 Manager 进行重新规划：

```python
# 错误升级检查
if len(info_pool.action_outcomes) >= err_to_manager_thresh:
    latest_outcomes = info_pool.action_outcomes[-err_to_manager_thresh:]
    if all(outcome in ["B", "C"] for outcome in latest_outcomes):
        info_pool.error_flag_plan = True  # 触发 Manager 重新规划
```

### 4.3 无效动作跳过

如果上一个动作是无效的（解析失败），跳过 Manager，直接让 Executor 重试：

```python
if not info_pool.error_flag_plan and len(info_pool.action_history) > 0:
    if info_pool.action_history[-1]['action'] == 'invalid':
        skip_manager = True
```

## 5. GUI-Owl 模型集成

`android_world_v3/android_world/agents/gui_owl.py` 是 GUI-Owl 模型在 AndroidWorld 环境中的具体实现。

### 5.1 核心 step() 方法

```python
class GUIOwlAgent(base_agent.EnvironmentInteractingAgent):
    def step(self, goal: str) -> AgentInteractionResult:
        # 1. 获取截图
        state = self.get_post_transition_state()
        screenshot = state.pixels.copy()
        
        # 2. 构建 prompt
        prompt = self._build_prompt(goal, self.action_history)
        
        # 3. 调用模型
        response = self.vllm.predict_mm(prompt, [screenshot_path])
        
        # 4. 解析动作
        action = self._parse_action(response)
        
        # 5. 执行动作
        self.env.execute_action(action)
        
        return AgentInteractionResult(done, info)
```

### 5.2 动作转换

```python
def convert_fc_action_to_json_action(dummy_action) -> JSONAction:
    action_json = json.loads(dummy_action)
    action_type = action_json['action']
    
    if action_type == 'click':
        return JSONAction(
            action_type=CLICK,
            x=action_json['coordinate'][0],
            y=action_json['coordinate'][1]
        )
    elif action_type == 'swipe':
        return JSONAction(
            action_type=SWIPE,
            direction=[start_x, start_y, end_x, end_y]
        )
    # ... 其他动作类型
```

## 6. 设备控制层

### 6.1 Android 控制器

```python
class AndroidController:
    def __init__(self, adb_path):
        self.adb_path = adb_path
        
    def get_screenshot(self, save_path):
        os.system(f"{self.adb_path} exec-out screencap -p > {save_path}")
        
    def tap(self, x, y):
        os.system(f"{self.adb_path} shell input tap {x} {y}")
        
    def swipe(self, x1, y1, x2, y2, duration=300):
        os.system(f"{self.adb_path} shell input swipe {x1} {y1} {x2} {y2} {duration}")
```

### 6.2 HarmonyOS 控制器

```python
class HarmonyOSController:
    def __init__(self, hdc_path):
        self.hdc_path = hdc_path
        
    def get_screenshot(self, save_path):
        os.system(f"{self.hdc_path} shell snapshot_display -f /data/local/tmp/screenshot.png")
        os.system(f"{self.hdc_path} file recv /data/local/tmp/screenshot.png {save_path}")
```

## 7. 动作空间定义

`new_json_action.py` 定义了所有原子动作常量：

| 动作常量 | 描述 |
|---|---|
| `CLICK` | 点击 |
| `LONG_PRESS` | 长按 |
| `SWIPE` | 滑动 |
| `TYPE` / `INPUT_TEXT` | 输入文本 |
| `NAVIGATE_HOME` | 返回主屏幕 |
| `NAVIGATE_BACK` | 返回上一页 |
| `OPEN_APP` | 打开应用 |
| `ANSWER` | 回答问题 |
| `TERMINATE` / `STATUS` | 结束任务 |

## 8. 使用示例

### 8.1 命令行运行

```bash
python run_mobileagentv3.py \
    --adb_path /path/to/adb \
    --api_key your_api_key \
    --base_url http://localhost:8000/v1 \
    --model GUI-Owl-32B \
    --instruction "打开微信，发送消息给张三" \
    --max_step 25
```

### 8.2 Python API 调用

```python
from utils.mobile_agent_e import InfoPool, Manager, Executor
from utils.call_mobile_agent_e import GUIOwlWrapper

# 初始化
vllm = GUIOwlWrapper(api_key, base_url, model)
info_pool = InfoPool(instruction="打开设置，开启蓝牙")

# 规划
manager = Manager()
plan_output = vllm.predict_mm(manager.get_prompt(info_pool), [screenshot])
info_pool.plan = manager.parse_response(plan_output)['plan']

# 执行
executor = Executor()
action_output = vllm.predict_mm(executor.get_prompt(info_pool), [screenshot])
action = executor.parse_response(action_output)['action']
```

## 9. 总结

Mobile-Agent v3 的多智能体协作架构是其核心创新：

| 智能体 | 职责 | 调用频率 |
|-------|------|---------|
| **Manager** | 高层规划、进度跟踪、错误升级处理 | 每步调用（除非跳过） |
| **Executor** | 选择和执行原子动作 | 每步调用 |
| **ActionReflector** | 验证动作结果、提供错误反馈 | 每步调用 |
| **Notetaker** | 记录重要信息 | 可选调用 |

这种架构的优势：
1. **职责分离**：每个智能体专注于特定任务，提高了系统的可维护性
2. **错误恢复**：多层次的错误处理机制，提高了任务成功率
3. **信息共享**：InfoPool 实现了跨智能体的状态共享
4. **灵活扩展**：可以轻松添加新的智能体或修改现有智能体的行为
