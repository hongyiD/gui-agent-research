# AppAgent 深度代码解构

> **作者**: Damon Li  
> **更新日期**: 2026年1月7日

## 一、项目结构

```
AppAgent/
├── learn.py                    # 探索阶段入口
├── run.py                      # 部署阶段入口
├── config.yaml                 # 配置文件
├── requirements.txt            # 依赖列表
├── assets/                     # 静态资源
│   ├── teaser.png             # 项目展示图
│   └── demo.png               # 演示图
└── scripts/                    # 核心脚本
    ├── __init__.py
    ├── and_controller.py       # Android 设备控制器
    ├── config.py               # 配置加载
    ├── document_generation.py  # 文档生成
    ├── model.py                # 模型接口
    ├── prompts.py              # 提示词模板
    ├── self_explorer.py        # 自主探索器
    ├── step_recorder.py        # 人类演示记录器
    ├── task_executor.py        # 任务执行器
    └── utils.py                # 工具函数
```

## 二、核心模块解析

### 2.1 任务执行器 (`task_executor.py`)

这是 AppAgent 部署阶段的核心文件，负责执行用户指定的任务。

#### 主循环逻辑

```python
while round_count < configs["MAX_ROUNDS"]:
    round_count += 1
    
    # 1. 获取屏幕截图和 UI 树
    screenshot_path = controller.get_screenshot(...)
    xml_path = controller.get_xml(...)
    
    # 2. 解析可交互元素
    if grid_on:
        # 网格模式
        rows, cols = draw_grid(screenshot_path, ...)
        prompt = prompts.task_template_grid
    else:
        # 标签模式
        clickable_list = []
        focusable_list = []
        traverse_tree(xml_path, clickable_list, "clickable", True)
        traverse_tree(xml_path, focusable_list, "focusable", True)
        
        # 合并元素列表，去除重复
        elem_list = clickable_list.copy()
        for elem in focusable_list:
            # 检查是否与已有元素过近
            if not close:
                elem_list.append(elem)
        
        # 绘制标签
        draw_bbox_multi(screenshot_path, ..., elem_list)
        
        # 加载 UI 文档
        if not no_doc:
            ui_doc = ""
            for i, elem in enumerate(elem_list):
                doc_path = os.path.join(docs_dir, f"{elem.uid}.txt")
                if os.path.exists(doc_path):
                    doc_content = ast.literal_eval(open(doc_path).read())
                    # 构建文档字符串
                    if doc_content["tap"]:
                        ui_doc += f"This UI element is clickable. {doc_content['tap']}\n"
                    # ... 其他动作类型
    
    # 3. 调用模型获取下一步动作
    status, rsp = mllm.get_model_response(prompt, [image])
    
    # 4. 解析并执行动作
    res = parse_explore_rsp(rsp)
    act_name = res[0]
    
    if act_name == "FINISH":
        task_complete = True
        break
    elif act_name == "tap":
        _, area = res
        tl, br = elem_list[area - 1].bbox
        x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
        controller.tap(x, y)
    elif act_name == "text":
        _, input_str = res
        controller.text(input_str)
    # ... 其他动作类型
```

#### 网格坐标转换

```python
def area_to_xy(area, subarea):
    """将网格区域和子区域转换为屏幕坐标"""
    area -= 1
    row, col = area // cols, area % cols
    x_0, y_0 = col * (width // cols), row * (height // rows)
    
    if subarea == "top-left":
        x, y = x_0 + (width // cols) // 4, y_0 + (height // rows) // 4
    elif subarea == "center":
        x, y = x_0 + (width // cols) // 2, y_0 + (height // rows) // 2
    # ... 其他子区域
    
    return x, y
```

### 2.2 自主探索器 (`self_explorer.py`)

自主探索模式的核心实现，包含探索和反思两个阶段。

#### 探索-反思循环

```python
while round_count < configs["MAX_ROUNDS"]:
    round_count += 1
    
    # 1. 探索阶段 - 获取截图并决定动作
    screenshot_before = controller.get_screenshot(f"{round_count}_before", task_dir)
    
    # 过滤已知无用的元素
    elem_list = []
    for elem in clickable_list:
        if elem.uid in useless_list:
            continue
        elem_list.append(elem)
    
    # 调用模型决定动作
    status, rsp = mllm.get_model_response(prompt, [base64_img_before])
    res = parse_explore_rsp(rsp)
    
    # 执行动作
    if act_name == "tap":
        controller.tap(x, y)
    # ...
    
    # 2. 反思阶段 - 评估动作效果
    screenshot_after = controller.get_screenshot(f"{round_count}_after", task_dir)
    
    # 调用模型进行反思
    status, rsp = mllm.get_model_response(prompt, [base64_img_before, base64_img_after])
    res = parse_reflect_rsp(rsp)
    decision = res[0]
    
    # 3. 根据反思结果处理
    if decision == "INEFFECTIVE":
        # 将元素加入无用列表
        useless_list.add(resource_id)
        last_act = "None"
    elif decision == "BACK" or decision == "CONTINUE" or decision == "SUCCESS":
        # 生成文档
        doc = res[-1]
        doc_content[act_name] = doc
        with open(doc_path, "w") as outfile:
            outfile.write(str(doc_content))
        doc_count += 1
```

#### 文档结构

```python
doc_content = {
    "tap": "",        # 点击功能描述
    "text": "",       # 文本输入用途
    "v_swipe": "",    # 垂直滑动功能
    "h_swipe": "",    # 水平滑动功能
    "long_press": ""  # 长按功能
}
```

### 2.3 Android 控制器 (`and_controller.py`)

封装 ADB 命令，提供设备控制接口。

#### 核心类

```python
class AndroidController:
    def __init__(self, device):
        self.device = device
    
    def get_device_size(self):
        """获取设备屏幕尺寸"""
        command = f"adb -s {self.device} shell wm size"
        # 解析输出获取 width, height
        return width, height
    
    def get_screenshot(self, prefix, save_dir):
        """获取屏幕截图"""
        cap_command = f"adb -s {self.device} shell screencap -p /sdcard/{prefix}.png"
        pull_command = f"adb -s {self.device} pull /sdcard/{prefix}.png {save_path}"
        # 执行命令
        return save_path
    
    def get_xml(self, prefix, save_dir):
        """获取 UI 层次结构 XML"""
        dump_command = f"adb -s {self.device} shell uiautomator dump /sdcard/{prefix}.xml"
        pull_command = f"adb -s {self.device} pull /sdcard/{prefix}.xml {save_path}"
        return save_path
    
    def tap(self, x, y):
        """点击指定坐标"""
        command = f"adb -s {self.device} shell input tap {x} {y}"
        return execute(command)
    
    def text(self, input_str):
        """输入文本"""
        input_str = input_str.replace(" ", "%s").replace("'", "")
        command = f"adb -s {self.device} shell input text '{input_str}'"
        return execute(command)
    
    def long_press(self, x, y, duration=1000):
        """长按"""
        command = f"adb -s {self.device} shell input swipe {x} {y} {x} {y} {duration}"
        return execute(command)
    
    def swipe(self, x, y, direction, dist="medium"):
        """滑动"""
        # 根据方向和距离计算终点坐标
        command = f"adb -s {self.device} shell input swipe {x} {y} {x2} {y2} {duration}"
        return execute(command)
    
    def back(self):
        """返回键"""
        command = f"adb -s {self.device} shell input keyevent KEYCODE_BACK"
        return execute(command)
```

#### UI 元素解析

```python
def traverse_tree(xml_path, elem_list, attrib, value):
    """遍历 UI 树，提取指定属性的元素"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for node in root.iter():
        if node.attrib.get(attrib) == str(value):
            bounds = node.attrib.get("bounds")
            # 解析边界框
            tl = (int(bounds[1:].split(",")[0]), int(bounds.split(",")[1].split("]")[0]))
            br = (int(bounds.split("[")[2].split(",")[0]), int(bounds.split(",")[2].split("]")[0]))
            
            elem = UIElement(
                uid=node.attrib.get("resource-id") or f"node_{len(elem_list)}",
                bbox=(tl, br),
                attrib=node.attrib
            )
            elem_list.append(elem)
```

### 2.4 模型接口 (`model.py`)

封装多模态模型调用接口。

#### OpenAI 模型

```python
class OpenAIModel:
    def __init__(self, base_url, api_key, model, temperature, max_tokens):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def get_model_response(self, prompt, images):
        """调用 GPT-4V 获取响应"""
        content = [{"type": "text", "text": prompt}]
        
        for image_path in images:
            # 将图片转为 base64
            base64_image = encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            })
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        )
        
        return True, response.json()["choices"][0]["message"]["content"]
```

#### Qwen 模型

```python
class QwenModel:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
    
    def get_model_response(self, prompt, images):
        """调用通义千问 VL 获取响应"""
        # 使用 DashScope SDK
        from dashscope import MultiModalConversation
        
        messages = [{
            "role": "user",
            "content": [
                {"image": images[0]},
                {"text": prompt}
            ]
        }]
        
        response = MultiModalConversation.call(
            model=self.model,
            messages=messages
        )
        
        return True, response.output.choices[0].message.content
```

#### 响应解析

```python
def parse_explore_rsp(rsp):
    """解析探索阶段的模型响应"""
    # 提取动作类型
    if "tap(" in rsp:
        match = re.search(r"tap\((\d+)\)", rsp)
        return ("tap", int(match.group(1)), rsp)
    elif "text(" in rsp:
        match = re.search(r"text\((.+?)\)", rsp)
        return ("text", match.group(1), rsp)
    elif "long_press(" in rsp:
        match = re.search(r"long_press\((\d+)\)", rsp)
        return ("long_press", int(match.group(1)), rsp)
    elif "swipe(" in rsp:
        match = re.search(r"swipe\((\d+),\s*\"(\w+)\",\s*\"(\w+)\"\)", rsp)
        return ("swipe", int(match.group(1)), match.group(2), match.group(3), rsp)
    elif "FINISH" in rsp:
        return ("FINISH", rsp)
    else:
        return ("ERROR", rsp)

def parse_reflect_rsp(rsp):
    """解析反思阶段的模型响应"""
    if "INEFFECTIVE" in rsp:
        return ("INEFFECTIVE",)
    elif "BACK" in rsp:
        doc = extract_doc(rsp)
        return ("BACK", doc)
    elif "CONTINUE" in rsp:
        doc = extract_doc(rsp)
        return ("CONTINUE", doc)
    elif "SUCCESS" in rsp:
        doc = extract_doc(rsp)
        return ("SUCCESS", doc)
    else:
        return ("ERROR",)
```

### 2.5 提示词模板 (`prompts.py`)

定义各阶段使用的提示词模板。

#### 任务执行提示词

```python
task_template = """You are a helpful AI assistant that can operate a smartphone.
Your task is: <task_description>

The screenshot shows the current state of the phone. Interactive elements are labeled with numeric tags.

<ui_document>

Your last action was: <last_act>

Based on the screenshot and the task, decide your next action. You can:
- tap(element_id): Tap on the element with the given numeric tag
- text(content): Input text content
- long_press(element_id): Long press on the element
- swipe(element_id, direction, distance): Swipe on the element
  - direction: "up", "down", "left", "right"
  - distance: "short", "medium", "long"
- grid: Switch to grid mode for more precise control
- FINISH: Task is completed

Output your action in the format: action(parameters)
"""
```

#### 自主探索提示词

```python
self_explore_task_template = """You are exploring a smartphone app to learn its functionality.
Your goal is: <task_description>

The screenshot shows the current state. Interactive elements are labeled with numeric tags.

Your last action was: <last_act>

Decide your next action to explore the app. Focus on elements that might help achieve the goal.
"""
```

#### 反思提示词

```python
self_explore_reflect_template = """You just performed <action> on element <ui_element>.

The first image shows the screen before the action.
The second image shows the screen after the action.

Task goal: <task_desc>
Your last action: <last_act>

Analyze the effect of your action and respond with one of:
- INEFFECTIVE: The action had no visible effect
- BACK: The action led to an unwanted state, need to go back
- CONTINUE: The action was useful, continue exploring
- SUCCESS: The action achieved or progressed toward the goal

If the action was useful (BACK, CONTINUE, or SUCCESS), also provide a brief documentation 
describing what this UI element does when <action> is performed.

Format: DECISION
Documentation: [description of the element's function]
"""
```

## 三、数据流图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AppAgent 数据流                                │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  用户    │────▶│   任务描述    │────▶│   主控制器   │────▶│  Android    │
│  输入    │     │   Task Desc  │     │   Main Loop  │     │  Controller │
└──────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                             │                    │
                                             ▼                    ▼
                                      ┌──────────────┐     ┌──────────────┐
                                      │   截图 +     │◀────│   ADB 命令   │
                                      │   XML 解析   │     │   执行       │
                                      └──────────────┘     └──────────────┘
                                             │
                                             ▼
                                      ┌──────────────┐
                                      │   元素标注   │
                                      │   + 文档加载 │
                                      └──────────────┘
                                             │
                                             ▼
                                      ┌──────────────┐
                                      │   构建提示词 │
                                      │   Prompt     │
                                      └──────────────┘
                                             │
                                             ▼
                                      ┌──────────────┐
                                      │   多模态模型 │
                                      │   GPT-4V /   │
                                      │   Qwen-VL    │
                                      └──────────────┘
                                             │
                                             ▼
                                      ┌──────────────┐
                                      │   响应解析   │
                                      │   Parse RSP  │
                                      └──────────────┘
                                             │
                                             ▼
                                      ┌──────────────┐
                                      │   动作执行   │────────────────────┐
                                      │   Execute    │                    │
                                      └──────────────┘                    │
                                             │                            │
                                             ▼                            ▼
                                      ┌──────────────┐            ┌──────────────┐
                                      │   循环继续   │            │   任务完成   │
                                      │   或结束     │            │   FINISH     │
                                      └──────────────┘            └──────────────┘
```

## 四、关键设计模式

### 4.1 知识积累模式

AppAgent 的核心创新是**知识积累**模式：

```
探索阶段                              部署阶段
┌─────────────────┐                  ┌─────────────────┐
│ 自主探索 / 演示  │                  │   执行任务      │
│                 │                  │                 │
│ ┌─────────────┐ │                  │ ┌─────────────┐ │
│ │ 尝试操作    │ │                  │ │ 加载文档    │ │
│ └─────────────┘ │                  │ └─────────────┘ │
│       ↓         │                  │       ↓         │
│ ┌─────────────┐ │                  │ ┌─────────────┐ │
│ │ 反思效果    │ │                  │ │ 精准操作    │ │
│ └─────────────┘ │                  │ └─────────────┘ │
│       ↓         │                  │                 │
│ ┌─────────────┐ │                  │                 │
│ │ 生成文档    │─┼─────────────────▶│                 │
│ └─────────────┘ │   Knowledge      │                 │
│                 │   Base           │                 │
└─────────────────┘                  └─────────────────┘
```

### 4.2 无用元素过滤

在自主探索中，AppAgent 维护一个 `useless_list`，记录已知无效的元素：

```python
useless_list = set()

# 探索时过滤
for elem in clickable_list:
    if elem.uid in useless_list:
        continue  # 跳过已知无用的元素
    elem_list.append(elem)

# 反思后更新
if decision == "INEFFECTIVE":
    useless_list.add(resource_id)
```

### 4.3 网格模式降级

当标签模式无法精确定位时，AppAgent 可以切换到网格模式：

```python
if grid_on:
    # 网格模式：将屏幕划分为 N×M 个区域
    rows, cols = draw_grid(screenshot_path, ...)
    prompt = prompts.task_template_grid
else:
    # 标签模式：使用数字标签标注元素
    draw_bbox_multi(screenshot_path, ..., elem_list)
    prompt = prompts.task_template
```

## 五、配置参数

```yaml
# config.yaml
MODEL: OpenAI                    # 模型提供商：OpenAI 或 Qwen
OPENAI_API_KEY: your_api_key     # OpenAI API 密钥
OPENAI_API_BASE: https://api.openai.com/v1  # API 基础 URL
OPENAI_API_MODEL: gpt-4-vision-preview      # 模型名称
DASHSCOPE_API_KEY: your_key      # 阿里云 API 密钥（Qwen 模式）
QWEN_MODEL: qwen-vl-max          # Qwen 模型名称

TEMPERATURE: 0.1                 # 生成温度
MAX_TOKENS: 500                  # 最大 token 数
REQUEST_INTERVAL: 3              # 请求间隔（秒）
MAX_ROUNDS: 20                   # 最大轮数
MIN_DIST: 30                     # 元素最小距离（像素）
DARK_MODE: false                 # 深色模式
```

## 六、扩展性

### 6.1 添加新模型

在 `model.py` 中添加新的模型类：

```python
class NewModel:
    def __init__(self, **kwargs):
        # 初始化
        pass
    
    def get_model_response(self, prompt, images):
        # 调用模型 API
        # 返回 (success, response)
        pass
```

### 6.2 添加新动作

1. 在 `and_controller.py` 中添加控制方法
2. 在 `prompts.py` 中更新提示词
3. 在 `model.py` 的解析函数中添加解析逻辑
4. 在 `task_executor.py` 中添加执行逻辑

## 七、总结

AppAgent 的代码架构清晰，核心创新在于：

1. **两阶段设计**：探索 + 部署，知识可复用
2. **反思机制**：自动评估动作效果，生成高质量文档
3. **网格模式**：突破 UI 元素限制，支持任意位置操作
4. **模块化设计**：易于扩展新模型和新动作

这种设计使得 AppAgent 在面对新应用时，可以通过探索阶段快速积累知识，在部署阶段高效执行任务。
