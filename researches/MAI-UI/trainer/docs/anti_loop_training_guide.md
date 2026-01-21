# GUI Agent 防循环训练指南

## 问题描述

GUI Agent 容易陷入死循环，表现为：
- 连续多次执行相同动作（如 swipe up）
- 每一步的 thinking 几乎相同
- 没有意识到"已经尝试很多次了，应该换策略"

## 根因分析

1. **训练数据缺乏负样本**：没有"失败后调整策略"的样本
2. **历史信息未被有效利用**：模型看不到或忽略了历史动作
3. **缺乏终止条件学习**：不知道什么时候应该放弃当前策略

## 解决方案

### 1. 构造防循环训练数据

需要在训练数据中添加以下类型的样本：

#### Type A: 循环检测 + ask_user

```json
{
  "prompt": "Task: 在微信吗喽群中找到灿旭关于胡宁的消息\n\nAction History:\n- Step 1: open 微信\n- Step 2: click 吗喽互助直面天命\n- Step 3: swipe up\n- Step 4: swipe up\n- Step 5: swipe up\n- Step 6: swipe up\n- Step 7: swipe up\n\nCurrent screenshot: [群聊界面，没有看到灿旭的消息]\n\nWhat should you do next?",
  "response": "<thinking>\n我已经连续向上滑动了5次，但仍然没有找到灿旭关于胡宁的消息。继续盲目滑动不是一个好策略，因为：\n1. 消息可能在更早的时间，需要滑动很多次\n2. 消息可能不存在\n3. 我可能漏看了\n\n此时应该询问用户获取更多信息，而不是继续无效滑动。\n</thinking>\n<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"ask_user\", \"text\": \"我已经向上滑动查看了多屏消息，但没有找到灿旭关于胡宁的发言。请问：1) 这条消息大概是什么时候发的？2) 灿旭在群里的昵称是否就是'灿旭'？3) 消息的大致内容是什么？\"}}\n</tool_call>"
}
```

#### Type B: 循环检测 + 换策略（使用搜索）

```json
{
  "prompt": "Task: 找到张三的聊天\n\nAction History:\n- Step 1: 在聊天列表 swipe down\n- Step 2: swipe down\n- Step 3: swipe down\n- Step 4: swipe down\n\nCurrent screenshot: [聊天列表，没有看到张三]\n\nWhat should you do next?",
  "response": "<thinking>\n我已经在聊天列表向下滑动了4次，但没有找到张三。继续滑动效率太低，我应该换一个策略：使用搜索功能。微信顶部有搜索框，可以直接搜索联系人名字。\n</thinking>\n<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"click\", \"coordinate\": [540, 120]}}\n</tool_call>"
}
```

#### Type C: 屏幕无变化检测 + 终止

```json
{
  "prompt": "Task: 查看更多历史消息\n\nAction History:\n- Step 5: swipe up\n- Step 6: swipe up (屏幕内容与Step 5相同)\n- Step 7: swipe up (屏幕内容与Step 6相同)\n\nCurrent screenshot: [与上一步完全相同的聊天界面]\n\nWhat should you do next?",
  "response": "<thinking>\n我注意到连续两次滑动后屏幕内容没有变化，这说明已经到达聊天记录的顶部，没有更多历史消息了。继续滑动是无效的。\n\n当前任务是查看更多历史消息，但已经没有更多了。我应该告知用户这个情况。\n</thinking>\n<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"answer\", \"text\": \"已经滑动到聊天记录的顶部，没有找到更多历史消息。\"}}\n</tool_call>"
}
```

#### Type D: 计数器意识

```json
{
  "prompt": "Task: 在抖音找到某个视频\n\nAction History:\n- Step 1-10: swipe up (连续刷了10个视频)\n\nCurrent screenshot: [抖音视频页面]\n\nWhat should you do next?",
  "response": "<thinking>\n我已经连续刷了10个视频，但仍然没有找到目标视频。盲目刷视频效率太低，而且抖音视频是无限流，可能永远刷不到。\n\n我应该使用搜索功能来找到特定视频，或者询问用户更多关于视频的信息（如创作者名称、视频标题关键词等）。\n</thinking>\n<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"ask_user\", \"text\": \"刷了10个视频没有找到目标。请提供更多信息：视频的创作者是谁？或者视频标题/内容有什么关键词？我可以使用搜索来查找。\"}}\n</tool_call>"
}
```

### 2. 训练数据格式要求

确保 prompt 中包含完整的 **Action History**：

```
Task: [任务描述]

Action History:
- Step 1: [动作1]
- Step 2: [动作2]
- Step 3: [动作3]
...

Current screenshot: [描述或图像]

What should you do next?
```

### 3. 关键训练信号

模型需要从训练数据中学会以下判断：

| 信号 | 应有的反应 |
|------|-----------|
| 连续 N 次相同动作（N≥3） | 考虑换策略或 ask_user |
| 屏幕内容连续 2 次无变化 | 认为已到达边界，停止当前动作 |
| 找不到目标 + 滑动超过 5 次 | 使用搜索功能 |
| 搜索也找不到 | ask_user 确认名称/关键词 |
| 点击无响应 + 重试 2 次 | 检查元素是否可点击，或等待加载 |

### 4. 数据增强策略

从现有轨迹数据生成防循环样本：

```python
def generate_anti_loop_samples(trajectory: list[dict]) -> list[dict]:
    """从正常轨迹生成防循环训练样本"""
    samples = []
    
    # 检测连续相同动作
    for i in range(len(trajectory) - 3):
        actions = [trajectory[j]["action"] for j in range(i, i + 4)]
        if len(set(actions)) == 1:  # 连续4个相同动作
            # 生成"应该换策略"的样本
            sample = {
                "prompt": build_prompt_with_history(trajectory[:i+4]),
                "response": generate_strategy_change_response(trajectory[i]["action"])
            }
            samples.append(sample)
    
    return samples
```

### 5. 推荐的训练数据比例

| 数据类型 | 比例 | 说明 |
|---------|------|------|
| 正常成功轨迹 | 60% | 学习正确的操作流程 |
| 循环检测样本 | 20% | 学习识别和打破循环 |
| 失败恢复样本 | 10% | 学习从错误中恢复 |
| ask_user 样本 | 10% | 学习何时求助用户 |

## 评估指标

训练后评估模型是否学会了防循环：

1. **循环检测率**：模型在连续 N 次相同动作后，是否会改变策略
2. **平均循环长度**：陷入循环时，平均多少步能跳出
3. **ask_user 调用率**：在合适的时机是否会求助用户

## 与 RL 的配合

如果 SFT 效果不够，可以用 RL 进一步优化：

```python
def compute_reward(trajectory: list[dict]) -> float:
    reward = 0
    
    # 惩罚重复动作
    for i in range(1, len(trajectory)):
        if trajectory[i]["action"] == trajectory[i-1]["action"]:
            reward -= 0.1  # 连续相同动作扣分
    
    # 惩罚长循环
    loop_length = detect_loop_length(trajectory)
    if loop_length > 3:
        reward -= 0.5 * loop_length
    
    # 奖励成功完成任务
    if trajectory[-1].get("success"):
        reward += 10
    
    # 奖励合理的 ask_user
    if appropriate_ask_user(trajectory):
        reward += 2
    
    return reward
```

## 使用方法

### 1. 数据增强

使用 `data_augmentation.py` 处理原始轨迹数据：

```bash
# 处理单个轨迹文件
python data_augmentation.py \
    -i ../dataset/20260119_201327/trajectory.jsonl \
    -o ../dataset/processed/sft_train_with_history.jsonl \
    --history-window 5 \
    --anti-loop-ratio 0.2

# 处理整个数据集目录
python data_augmentation.py \
    -i ../dataset/ \
    -o ../dataset/processed/sft_train_all.jsonl \
    --history-window 5 \
    --anti-loop-ratio 0.2
```

### 2. 更新训练配置

修改 `my_config_119.yaml`：

```yaml
data:
  # 使用增强后的数据
  processed_data_file: "../../dataset/processed/sft_train_with_history.jsonl"
```

### 3. 推理时启用循环检测

在推理代码中集成 `LoopDetector`：

```python
from loop_detector import LoopDetector, LoopDetectorConfig, integrate_with_agent

# 初始化检测器
config = LoopDetectorConfig(
    same_action_threshold=3,  # 连续3次相同动作触发
    cycle_detection_window=10,
)
detector = LoopDetector(config)

# 在每次获取模型输出后
def agent_step(model, screenshot):
    # 模型推理
    raw_output = model.predict(screenshot)
    
    # 循环检测（可能替换输出）
    final_output = integrate_with_agent(raw_output, detector)
    
    return final_output
```

## 完整训练流程

```
原始轨迹数据
    │
    ▼
[data_augmentation.py] ──→ 添加 Action History
    │                      生成 Anti-Loop 样本
    ▼
增强后的训练数据
    │
    ▼
[sft_trainer.py] ──→ SFT 训练
    │
    ▼
微调后的模型
    │
    ▼
[推理 + loop_detector.py] ──→ 运行时循环检测（兜底）
```

## 总结

防循环能力需要通过以下方式获得：

1. **SFT 阶段**：构造包含历史动作的训练数据，让模型学会"看历史、判循环、换策略"
2. **推理阶段**：使用 `LoopDetector` 作为运行时兜底
3. **RL 阶段（可选）**：通过奖惩机制强化防循环行为

## 预期效果

| 指标 | 优化前 | 优化后（预期） |
|------|-------|---------------|
| 平均循环长度 | 10+ 步 | 3-4 步 |
| ask_user 触发率 | ~0% | 10-20% |
| 任务成功率 | 低（循环导致失败） | 提升 |
