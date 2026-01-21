# AgentCPM-GUI LoRA 微调实战技术笔记

## 1. 核心微调配置 (SFT LoRA)
AgentCPM-GUI 基于 MiniCPM-V 架构，其 LoRA 微调不仅涉及 LLM 部分，还需关注视觉编码器的协同。

### 1.1 目标模块 (Target Modules)
官方代码库中定义的 LoRA 目标模块采用正则表达式匹配，重点在于 LLM 的注意力层：
- **匹配正则**: `llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)`
- **关键说明**: 默认不包含 MLP (dense) 层。如果需要增强拟合能力，建议手动扩展至 `(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)`。

### 1.2 关键参数配置
- **秩与缩放**: 默认 `lora_r=64`, `lora_alpha=64`。在 GUI 复杂任务中，高秩有助于捕捉精细的坐标映射关系。
- **可训练模块保存**: `modules_to_save` 必须包含 `embed_tokens` 和 `resampler`。如果开启视觉微调，还需包含 `vpm`。
- **学习率**: 官方脚本 `finetune_lora.sh` 设置为 `1e-6`。注意：LoRA 通常建议 `1e-4` 级别，但 AgentCPM-GUI 官方采用极低学习率以维持预训练的 Grounding 能力。

## 2. 视觉与语言的调优策略
针对“仅微调语言部分”的问题，实测与代码逻辑结论如下：

### 2.1 联合微调逻辑
- **配置**: `tune_vision=true` + `use_lora=true`。
- **原理**: 视觉编码器 (`vpm`) 负责提取 GUI 元素的空间特征，Resampler 将特征对齐至 LLM。GUI 任务的核心是坐标预测，冻结视觉层会导致 LLM 无法修正因分辨率或系统风格差异带来的特征偏差。

### 2.2 仅调语言层的收益评估
- **适用场景**: 仅当任务指令极其复杂但界面分布与预训练集完全一致时。
- **局限性**: 收益极低（通常 < 15% 成功率提升）。因为 GUI Agent 的“思维”是建立在“视觉感官”之上的，视觉层不更新，LLM 很难凭空优化坐标精度。

## 3. RFT (强化微调) 深度细节
AgentCPM-GUI 引入 RFT (GRPO 算法) 提升逻辑推理能力（“先思考再行动”）。

### 3.1 奖励函数 (Reward Functions)
官方实现了多维度奖励检测，确保输出符合 GUI 操作规范：
- **schema**: 强制 JSON 格式符合定义的 Schema。
- **type**: 验证 `ActionType` 是否在合法枚举范围内。
- **args**: 验证动作参数（如坐标是否在 0-1000 间，`duration` 是否合法）。
- **react**: 核心奖励，验证“思维链 (thought)”与“动作 (action)”的一致性。

## 4. 训练实战建议
### 4.1 数据量要求
- **基础指令适配**: 至少 500+ 条高质量轨迹。
- **垂直领域/新系统**: 建议 2000-5000 条轨迹，且必须开启 `tune_vision`。

### 4.2 训练环境
- **显存优化**: 必须开启 `gradient_checkpointing: true`。
- **推理一致性**: 系统提示词 (System Prompt) 必须包含完整的 `ACTION_SCHEMA` 且开启 `thought` 字段。

## 5. 对比评估 (Doubao.md 准确性辨析)
- **模块误区**: Doubao 建议的 `[q_proj, k_proj, v_proj, o_proj, dense]` 与官方正则匹配模式不符，且漏掉了关键的 `modules_to_save` (resampler)。
- **LR 误区**: Doubao 推荐 `2e-4`，而官方生产脚本使用 `1e-6`。过大的 LR 在此模型上极易导致 Grounding 能力崩溃。
- **数据格式**: 官方 SFT 依赖特定对话格式，而非 Doubao 提供的简化 JSON 结构。建议参考 `dataset.py` 中的 `SupervisedDataset` 实现。
