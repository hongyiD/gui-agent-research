# 训练样本格式说明

本文档详细说明了一个conversation的完整训练样本格式。

## 1. 原始数据格式 (JSONL文件)

训练数据文件是JSONL格式，每行是一个独立的训练样本。每个样本包含以下结构：

```json
{
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.\n\n## Output Format\n..."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "<image_path>../dataset/20260119_201327/step_025.png</image_path>"
        },
        {
          "type": "text",
          "text": "Task: 帮我打开 139 邮箱...\n\nPrevious actions:\n- Thought: I will open the contact...\n  Action: {...}"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "<thinking>\nThought: I will send the composed follow-up message to 谭文辉.\n</thinking>\n<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"click\", \"coordinate\": [614, 495]}}\n</tool_call>"
        }
      ]
    }
  ],
  "metadata": {
    "source": "../dataset/20260119_201327/trajectory.jsonl",
    "step_index": 25,
    "action_type": "click",
    "success": true,
    "timestamp": "2026-01-19T20:16:10.912318",
    "screenshot_path": "../dataset/20260119_201327/step_025.png"
  }
}
```

### 字段说明

#### messages 字段
- **格式**: `openai_messages` 格式，兼容OpenAI API
- **结构**: 包含3个角色的消息
  - `system`: 系统提示词，定义agent的角色和行为规则
  - `user`: 用户输入，包含：
    - 图片路径或base64编码的图片（通过 `<image_path>...</image_path>` 或 `<image_base64>...</image_base64>` 标签）
    - 任务描述和之前的动作历史
  - `assistant`: 模型期望的输出，包含思考过程和动作

#### content 字段
- **多模态格式**: `content` 是一个列表，每个元素可以是：
  - `{"type": "text", "text": "..."}`: 文本内容
  - `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`: base64编码的图片
  - `{"type": "image", "image": <PIL.Image>}`: PIL图片对象（训练时转换后）

#### metadata 字段
- `source`: 原始轨迹文件路径
- `step_index`: 步骤索引
- `action_type`: 动作类型（click, type, swipe等）
- `success`: 动作是否成功
- `timestamp`: 时间戳
- `screenshot_path`: 截图路径
- `image_width`: 图像宽度（像素值，用于坐标验证）
- `image_height`: 图像高度（像素值，用于坐标验证）

### 坐标系统说明

**重要**：训练数据中的坐标使用**绝对坐标（像素值）**格式，与模型输出格式一致。

- **Assistant content 中的坐标**：绝对坐标（像素值），如 `[614, 495]`
- **历史动作中的坐标**：绝对坐标（像素值），如 `[279, 472]`
- **坐标转换**：如果轨迹数据中是归一化坐标（0-1范围），`data_processor.py` 会自动转换为绝对坐标

**坐标格式示例**：
```json
{
  "action": "click",
  "coordinate": [614, 495]  // 绝对坐标（像素值），不是归一化坐标
}
```

**推理时的处理流程**：
1. 模型输出：绝对坐标（像素值），如 `[614, 495]`
2. `parse_action_to_structure_output()`: 将绝对坐标除以 SCALE_FACTOR (999) 转换为标准化坐标（0-1范围）
3. 执行时：将标准化坐标乘以图像尺寸，转换回绝对坐标执行

## 2. 训练时的转换流程

### 2.1 数据加载阶段 (`PreprocessedMultiModalDataset`)

```python
# 1. 从JSONL文件加载原始数据
raw_data = []
with open(data_path, "r") as f:
    for line in f:
        raw_data.append(json.loads(line))

# 2. 预处理每个样本
for sample in raw_data:
    # 提取图片（从base64或路径）
    images = extract_images_from_messages(sample["messages"])
    
    # 限制图片数量（默认最多3张，保留最后N张）
    if len(images) > max_images_per_sample:
        images = images[-max_images_per_sample:]
    
    # 缓存到内存
    preprocessed_data.append({
        "sample": sample,
        "images": images  # 已解码的PIL Image对象
    })
```

### 2.2 批处理阶段 (`MultiModalDataCollator`)

```python
# 1. 转换为Qwen VL格式
qwen_messages = []
for msg in messages:
    role = msg["role"]
    content = msg["content"]
    
    if isinstance(content, list):
        qwen_content = []
        for item in content:
            if item["type"] == "text":
                qwen_content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image_url":
                # 替换为PIL Image对象
                qwen_content.append({"type": "image", "image": images[image_idx]})
                image_idx += 1
        qwen_messages.append({"role": role, "content": qwen_content})

# 2. 使用processor处理
text = processor.apply_chat_template(
    qwen_messages,
    tokenize=False,
    add_generation_prompt=False,
)

inputs = processor(
    text=[text],
    images=images,
    padding=False,
    return_tensors="pt",
)

# 3. 创建labels（mask prompt部分）
labels = create_labels(input_ids, qwen_messages)
```

### 2.3 Qwen VL格式示例

转换后的Qwen VL格式：

```python
[
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a GUI agent..."}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Task: 帮我打开 139 邮箱...\n\nPrevious actions:..."},
            {"type": "image", "image": <PIL.Image对象>}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Thought: I will send...\nAction: {...}"}
        ]
    }
]
```

## 3. 最终输入模型的格式

经过 `MultiModalDataCollator` 处理后，每个batch包含：

```python
{
    "input_ids": torch.Tensor,          # [batch_size, seq_len] - token化的文本序列
    "attention_mask": torch.Tensor,     # [batch_size, seq_len] - 注意力掩码
    "labels": torch.Tensor,              # [batch_size, seq_len] - 训练标签（-100表示不计算loss）
    "pixel_values": torch.Tensor,       # [batch_size, num_images, C, H, W] - 图片像素值
    "image_grid_thw": torch.Tensor,     # [batch_size, num_images, 3] - 图片网格信息
}
```

### Labels的创建规则

- **Prompt部分**（system + user消息）: `labels = -100`（不计算loss）
- **Response部分**（assistant消息）: `labels = input_ids`（计算loss）

这样确保模型只学习生成正确的response，而不学习重复prompt。

## 4. 一个完整样本的示例

### 4.1 System消息
```
You are a GUI agent. You are given a task and your action history, with screenshots. 
You need to perform the next action to complete the task.

## Output Format
For each function call, return the thinking process in <thinking> </thinking> tags, 
and a json object with function name and arguments within <tool_call></tool_call> XML tags.

## Action Space
{"action": "click", "coordinate": [x, y]}
{"action": "type", "text": "..."}
...
```

### 4.2 User消息
```
<image_path>../dataset/20260119_201327/step_025.png</image_path>

Task: 帮我打开 139 邮箱，搜索标题包含"2025年Q4销售数据"的邮件，下载其中的 Excel 附件。
退出139邮箱app，然后打开移动云盘app，搜索并进入"销售部-部门共享"文件夹，
把刚才下载的 Excel 文件上传进去...

Previous actions:
- Thought: I will open the contact 谭文辉 to send the copied link and message.
  Action: {"action": "click", "coordinate": [279, 472]}
- Thought: I will paste the copied link into the input box.
  Action: {"action": "long_press", "coordinate": [268, 952]}
...
```

### 4.3 Assistant消息（期望输出）

**格式**：使用 `<thinking>` 和 `<tool_call>` 标签

```
<thinking>
Thought: I will send the composed follow-up message to 谭文辉.
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [614, 495]}}
</tool_call>
```

**注意**：
- 坐标是**绝对坐标（像素值）**，如 `[614, 495]`，不是归一化坐标（0-1范围）
- 坐标值应该在图像尺寸范围内（可通过 metadata 中的 `image_width` 和 `image_height` 验证）

## 5. 数据格式支持

训练器支持三种数据格式，会自动检测并转换：

1. **`openai_messages`** (当前使用的格式)
   ```json
   {"messages": [...], "metadata": {...}}
   ```

2. **`prompt_response`**
   ```json
   {"prompt": "...", "response": "...", "metadata": {...}}
   ```

3. **`full_trajectory`**
   ```json
   {"task_goal": "...", "steps": [...], "metadata": {...}}
   ```
   会自动转换为多个 `prompt_response` 样本（每个step一个样本）

## 6. 查看实际样本

使用提供的脚本查看实际样本：

```bash
python scripts/show_training_sample.py <数据文件路径> [样本索引]
```

示例：
```bash
python scripts/show_training_sample.py ../dataset/processed/sft_train_final.jsonl 0
```

## 7. 坐标系统详细说明

### 7.1 训练数据坐标格式

训练数据使用**绝对坐标（像素值）**格式，例如：
- 图像尺寸：1080 x 2400 像素
- 坐标值：`[614, 495]` 表示图像中第 614 列、第 495 行的像素位置

### 7.2 坐标转换流程

**数据预处理阶段**（`data_processor.py`）：
1. 如果轨迹数据中是归一化坐标（0-1范围），自动转换为绝对坐标
2. 需要图像尺寸信息才能进行转换
3. 转换后的坐标保存在训练数据的 assistant content 中

**推理阶段**（`mai_naivigation_agent.py`）：
1. 模型输出：绝对坐标（像素值），如 `[614, 495]`
2. `parse_action_to_structure_output()`: 除以 SCALE_FACTOR (999) 转换为标准化坐标（0-1范围）
3. 执行时：将标准化坐标乘以图像尺寸，转换回绝对坐标执行

### 7.3 验证坐标格式

使用 `show_training_sample.py` 脚本可以验证训练数据中的坐标格式：
- 检查坐标是否为绝对坐标（像素值）
- 验证坐标是否在图像尺寸范围内
- 如果发现归一化坐标，会显示警告和建议

## 8. 关键代码位置

- **数据预处理**: `data/data_processor.py`
  - 坐标转换: `convert_coordinates_in_action()` (第 189-223 行)
  - 图像尺寸获取: `get_image_size()` (第 226-240 行)
  - 训练样本构建: `build_training_sample()` (第 377-495 行)
- **训练代码**: `sft_trainer.py`
  - **数据加载**: `sft_trainer.py:1564-1575`
  - **格式检测和转换**: `sft_trainer.py:613-697`
  - **图片提取**: `sft_trainer.py:373-462`
  - **Qwen格式转换**: `sft_trainer.py:1081-1113`
  - **批处理**: `sft_trainer.py:904-1257`
