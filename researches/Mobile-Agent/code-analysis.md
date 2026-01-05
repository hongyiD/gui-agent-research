
# Mobile-Agent 深度代码解构

> **作者**: Damon Li  
> **更新日期**: 2026年1月

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

## 2. 核心工作流解析 (`run_mobileagentv3.py`)

`mobile_v3/run_mobileagentv3.py` 是在真实设备上启动 Mobile-Agent-v3 的入口脚本。其核心逻辑是一个循环，在每个步骤中依次调用 Manager、Executor 和 Reflector。

```python
# run_mobileagentv3.py

def run_instruction(...):
    # 1. 初始化设备控制器 (Android/HarmonyOS)
    controller = AndroidController(adb_path) 

    # 2. 初始化信息池 (InfoPool) 和四个智能体
    info_pool = InfoPool(...)
    vllm = GUIOwlWrapper(api_key, base_url, model)
    manager = Manager()
    executor = Executor()
    notetaker = Notetaker()
    action_reflector = ActionReflector()

    # 3. 主循环，最多执行 max_step 次
    for step in range(max_step):
        # 3.1 获取当前屏幕截图
        controller.get_screenshot(...)

        # 3.2 调用 Manager 进行规划
        # 如果连续出错，会触发重新规划
        prompt_planning = manager.get_prompt(info_pool)
        output_planning, ... = vllm.predict_mm(prompt_planning, [screenshot])
        parsed_result_planning = manager.parse_response(output_planning)
        info_pool.plan = parsed_result_planning['plan']

        # 如果计划完成，则退出
        if "Finished" in info_pool.plan:
            break

        # 3.3 调用 Executor 生成具体动作
        prompt_action = executor.get_prompt(info_pool)
        output_action, ... = vllm.predict_mm(prompt_action, [screenshot])
        parsed_result_action = executor.parse_response(output_action)
        action_object = json.loads(parsed_result_action['action'])

        # 3.4 执行动作 (click, swipe, type, ...)
        controller.tap(action_object['coordinate'][0], ...)

        # 3.5 获取动作执行后的屏幕截图
        controller.get_screenshot(...)

        # 3.6 调用 Action Reflector 进行反思
        prompt_action_reflect = action_reflector.get_prompt(info_pool)
        output_action_reflect, ... = vllm.predict_mm(prompt_action_reflect, [before_screenshot, after_screenshot])
        parsed_result_action_reflect = action_reflector.parse_response(output_action_reflect)
        outcome = parsed_result_action_reflect['outcome'] # "A", "B", or "C"

        # 3.7 如果动作成功，调用 Notetaker 记录关键信息
        if outcome == "A" and if_notetaker:
            prompt_note = notetaker.get_prompt(info_pool)
            output_note, ... = vllm.predict_mm(prompt_note, [after_screenshot])
            ...
```

## 3. 多智能体框架解析 (`mobile_agent_e.py`)

`mobile_v3/utils/mobile_agent_e.py` 定义了 Mobile-Agent-v3 的核心多智能体框架。

### 3.1 `InfoPool` 数据类

`InfoPool` 是一个贯穿所有智能体的数据中心，用于存储和共享任务状态和历史信息。

-   **用户输入**: `instruction`
-   **工作记忆**: `summary_history`, `action_history`, `action_outcomes`
-   **规划信息**: `plan`, `completed_plan`
-   **长时记忆**: `important_notes`

### 3.2 `Manager` (管理者)

-   **`get_prompt()`**: 构建用于规划的 prompt。首次调用时，它要求模型制定一个高层计划。后续调用时，它会提供历史操作、当前计划、上次动作和结果，要求模型评估并更新计划。
-   **`parse_response()`**: 从模型的响应中解析出 `thought`（思考过程）、`completed_subgoal`（已完成的子目标）和 `plan`（更新后的计划）。

### 3.3 `Executor` (执行者)

-   **`get_prompt()`**: 构建用于生成具体动作的 prompt。它向模型提供用户请求、整体计划、当前子目标、历史动作和结果，要求模型选择一个原子动作。
-   **`parse_response()`**: 从模型的响应中解析出 `thought`、`action`（JSON 格式的动作）和 `description`（动作描述）。

### 3.4 `ActionReflector` (动作反思者)

-   **`get_prompt()`**: 构建用于反思的 prompt。它向模型提供用户请求、上次执行的动作、期望的结果，以及动作前后的两张屏幕截图。
-   **`parse_response()`**: 从模型的响应中解析出 `outcome`（结果，A/B/C 三种状态）和 `error_description`（如果失败，提供错误描述）。

### 3.5 `Notetaker` (记录员)

-   **`get_prompt()`**: 构建用于记录笔记的 prompt。它向模型提供用户请求、当前进度和已有的笔记，要求模型从当前屏幕中提取并更新重要信息。
-   **`parse_response()`**: 从模型的响应中解析出更新后的 `important_notes`。

## 4. 核心模型实现 (`gui_owl.py`)

`android_world_v3/android_world/agents/gui_owl.py` 是 GUI-Owl 模型在 AndroidWorld 环境中的具体实现。它继承自 `base_agent.EnvironmentInteractingAgent`。

-   **`step()` 方法**: 这是智能体与环境交互的核心。它接收 `goal` 作为输入，然后执行以下操作：
    1.  获取当前屏幕截图和 UI 元素。
    2.  构建适用于 GUI-Owl 模型的 prompt，这个 prompt 包含了任务目标、历史动作和当前截图。
    3.  调用 VLLM 服务（`self.vllm.predict_mm`）获取模型的动作预测。
    4.  解析模型返回的动作（一个包含 `action` 和 `arguments` 的 JSON 对象）。
    5.  将模型生成的动作转换为环境可以执行的 `JSONAction`。
    6.  通过 `actuation.execute_adb_action` 在设备上执行该动作。
    7.  记录动作、思考过程和总结，为下一步做准备。

-   **动作转换**: `mobile_agent_utils.convert_mobile_agent_action_to_json_action` 函数负责将 GUI-Owl 输出的动作格式（如 `{'action': 'click', 'target': 'button "OK"'}`）转换为 AndroidWorld 环境可以理解的、基于坐标的 `JSONAction`。

## 5. 动作空间定义 (`new_json_action.py`)

`android_world_v3/android_world/agents/new_json_action.py` 定义了智能体可以执行的所有原子动作的常量。

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

## 6. 总结

Mobile-Agent-v3 的代码实现清晰地展示了其多智能体协作的架构。通过将任务分解为规划、执行、反思和记录等多个环节，并为每个环节设计专门的智能体和 prompt，该框架显著提升了在复杂 GUI 任务上的表现。其核心驱动力是强大的 GUI-Owl 模型，而 `run_mobileagentv3.py` 中的主循环则是整个框架的“指挥中心”，协调各个智能体完成用户指令。
