# Qwen3-VL LoRA 微调技术说明

本文档说明在对 Qwen3-VL / MAI-UI 等视觉语言模型进行 LoRA 微调时的关键技术细节。

## 1. 训练警告信息解读

在启动训练时，可能会出现以下警告信息，均为正常现象：

| 警告信息 | 说明 | 影响 |
|---------|------|------|
| `FutureWarning: tokenizer is deprecated` | Transformers 5.0 版本后将移除 `tokenizer` 参数，建议改用 `processing_class` | 当前无影响，将来升级库时需修改代码 |
| `kernel version 3.10.0 below 5.5.0` | Linux 内核版本较老（CentOS 7 默认），PyTorch 建议更高版本 | 极端情况下可能导致进程挂起，大多数情况正常 |
| `model already on multiple devices` | 模型已分布到多卡，跳过额外的设备移动 | 正常行为，多卡训练符合预期 |
| `tokenizer has new PAD/BOS/EOS tokens` | tokenizer 和模型配置的特殊 token 不一致，已自动对齐 | 自动修复，无需担心 |
| `use_cache=True incompatible with gradient checkpointing` | 梯度检查点和 KV cache 互斥，自动关闭缓存 | 正常行为，梯度检查点用于节省显存 |

### 1.1 关于 Tokenizer PAD/BOS/EOS Tokens 对齐的详细说明

训练时可能看到如下提示：

```
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. 
The model config and generation config were aligned accordingly, being updated with the tokenizer's values. 
Updated tokens: {'bos_token_id': None}.
```

**这是正常行为，不需要担心。** 原因如下：

1. **Qwen 系列模型不使用传统 BOS token**
   
   Qwen/Qwen2/Qwen3 模型采用 **ChatML 格式**，使用 `<|im_start|>` 和 `<|im_end|>` 标记对话边界，而不是传统的 BOS（Beginning of Sentence）token：
   
   ```
   <|im_start|>user
   你好<|im_end|>
   <|im_start|>assistant
   你好！<|im_end|>
   ```

2. **`bos_token_id: None` 是设计如此**
   
   因为 Qwen 使用 ChatML 格式，不需要单独的 BOS token，所以 `bos_token_id = None` 是正确的配置。

3. **系统自动对齐配置**
   
   Transformers 库检测到 tokenizer 和模型配置不一致时，会自动将模型配置更新为 tokenizer 的值，确保一致性。这是一个**自动修复**的过程，不需要手动干预。

**结论**：这些警告不会阻止训练运行，可以正常继续训练。

---

## 2. LoRA 微调范围说明

### 2.1 当前配置的 target_modules

```yaml
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

这些模块均属于 **语言模型（LLM）部分**：

- `q_proj / k_proj / v_proj / o_proj` → Self-Attention 层
- `gate_proj / up_proj / down_proj` → Feed-Forward Network (FFN) 层

### 2.2 Qwen3-VL 模型架构

```
Qwen3-VL / MAI-UI
│
├── Visual Encoder (视觉编码器)
│   ├── Patch Embedding
│   ├── Transformer Blocks
│   │   ├── Self-Attention (qkv, proj)
│   │   └── MLP (fc1, fc2)
│   └── ...
│
├── Projector (视觉-语言投影层)
│   └── 将视觉特征映射到 LLM 的 embedding 空间
│
└── Language Model (语言模型)
    ├── Embedding Layer
    ├── Transformer Blocks
    │   ├── Self-Attention (q_proj, k_proj, v_proj, o_proj) ← LoRA 作用于此
    │   └── FFN (gate_proj, up_proj, down_proj) ← LoRA 作用于此
    └── LM Head
```

### 2.3 当前配置的训练范围

| 模块 | 是否被 LoRA 微调 |
|------|-----------------|
| Visual Encoder (视觉编码器) | ❌ 不训练 |
| Projector (投影层) | ❌ 不训练 |
| Language Model (语言模型) | ✅ 训练 |

**结论**：当前配置只微调语言模型部分的参数，视觉编码器的参数保持冻结。

---

## 3. 是否需要微调视觉部分？

### 3.1 场景分析

| 场景 | 建议 | 理由 |
|------|------|------|
| 学习新的输出格式/指令遵循 | 只调 LLM | 视觉理解能力已足够，只需调整输出行为 |
| 适配特定 UI 风格/元素 | 考虑调视觉部分 | 可能需要增强对特定视觉特征的理解 |
| 数据集与预训练领域差异大 | 建议调视觉部分 | 需要学习新的视觉概念 |
| 显存有限 | 只调 LLM | 视觉编码器参数量大，加入后显存占用增加 |

### 3.2 只调 LLM 的优点

1. **显存占用更低**：视觉编码器参数量大，不训练可节省大量显存
2. **训练更稳定**：保持预训练的视觉理解能力
3. **收敛更快**：可训练参数更少，优化更高效
4. **泛化性更好**：避免在小数据集上过拟合视觉特征

### 3.3 调整视觉部分的优点

1. **更强的领域适应**：针对特定视觉内容进行优化
2. **更好的细粒度理解**：学习识别特定的 UI 元素、图标等
3. **端到端优化**：整个模型协同优化

---

## 4. 图片分辨率对 SFT-LoRA 微调的影响

### 4.1 核心结论

**关键区分：训练过程 vs 训练效果**

当视觉编码器冻结、只训练 LLM 的 attention 层（如 `q_proj, k_proj, v_proj, o_proj`）时：

| 方面 | 是否受影响 | 说明 |
|------|-----------|------|
| **训练过程** | ❌ 不受影响 | 训练可以正常进行，loss 可以正常下降，梯度可以正常计算 |
| **训练效果** | ✅ 会受影响 | LLM 学习的是"如何利用给定质量的视觉特征"，特征质量由输入分辨率决定 |

### 4.2 影响机制

数据流：

```
图片 (低分辨率，如 520×520) 
  ↓
视觉编码器（冻结，只前向传播）
  ↓
视觉特征（低质量，因为分辨率低）
  ↓
Projector（冻结）
  ↓
LLM Embedding（包含低质量视觉信息）
  ↓
LLM Attention（LoRA训练）← 学习如何利用这些特征
  ↓
输出
```

**关键理解**：
- 视觉编码器冻结时，分辨率低只是让特征提取更快（处理更少像素）
- 但 LLM 学习的是"如何利用低质量特征"
- 如果特征质量低，LLM 学到的模式也是基于低质量特征的

### 4.3 分辨率过低对特征质量的影响

#### 4.3.1 空间分辨率下降

Qwen3-VL 视觉编码器基于 Vision Transformer (ViT)：
- Patch Size: 16 像素
- Merge Size: 2
- 有效 Patch 大小: 16 × 2 = 32 像素

量化示例：

| 原始尺寸 | Resize到520 | Resize到1024 | Patch数量对比 |
|---------|------------|-------------|--------------|
| 1080×2400 | 234×520 | 460×1024 | - |
| Patches (520) | ~7×16 = 112 | - | - |
| Patches (1024) | - | ~29×32 = 928 | 8.3倍 |

影响：
- **空间细节丢失**：每个 patch 覆盖的物理区域变大
- **特征粒度变粗**：无法区分相近的小元素
- **边界模糊**：元素边界可能跨多个 patch

#### 4.3.2 文字识别能力下降

| 分辨率 | 文字大小 | 可读性 | 特征质量 |
|--------|---------|--------|---------|
| 1024 | ~24px | 清晰 | 每个字符有足够像素 |
| 520 | ~12px | 模糊 | 字符可能跨多个patch，特征混乱 |
| 256 | ~6px | 不可读 | 字符信息丢失 |

#### 4.3.3 图标/按钮识别精度下降

GUI 元素识别示例：

```
原始: 1080×2400
├─ 大按钮: 200×80px → 特征清晰
├─ 中图标: 80×80px → 特征可识别
└─ 小图标: 40×40px → 特征模糊

Resize到520: 234×520
├─ 大按钮: 43×17px → 特征仍可识别（约2个patch）
├─ 中图标: 17×17px → 特征模糊（<1个patch）
└─ 小图标: 9×9px → 特征丢失（<0.5个patch）
```

#### 4.3.4 空间定位精度下降

Grounding 任务的影响：

| 分辨率 | 坐标精度 | 示例 |
|--------|---------|------|
| 1024 | ±2-4像素 | 可以精确定位按钮中心 |
| 520 | ±8-16像素 | 可能定位到按钮边缘 |
| 256 | ±32-64像素 | 可能定位到错误元素 |

### 4.4 训练和推理分辨率一致性原则

**核心原则：训练和推理分辨率应保持一致**

#### 场景 1：训练和推理都用低分辨率（520）

```
训练阶段：
- 图片: 520×520
- 视觉特征: 低质量（小文字、小图标信息丢失）
- LLM学习: "如何从低质量特征中识别元素"
- 效果: 可以工作，但准确率受限

推理阶段：
- 图片: 520×520（与训练一致）
- 效果: 与训练时一致，可以接受
```

**结论**：如果训练和推理都用相同分辨率，影响较小。

#### 场景 2：训练用低分辨率（520），推理用高分辨率（1024）

```
训练阶段：
- 图片: 520×520
- 视觉特征: 低质量
- LLM学习: "如何从低质量特征中识别元素"
  - 学习到: "模糊的红色区域 = 139邮箱图标"
  - 学习到: "无法识别小文字"

推理阶段：
- 图片: 1024×1024（高分辨率）
- 视觉特征: 高质量（包含小文字、细节）
- LLM期望: 高质量特征（但训练时没见过）
- 效果: 可能表现不佳，因为特征分布不匹配
```

**结论**：训练和推理分辨率不一致可能导致性能下降。

### 4.5 分辨率选择建议

#### 策略 1：训练和推理保持一致（推荐）

```yaml
# 训练时
max_dim: 520

# 推理时也要用相同分辨率
max_dim: 520  # 保持一致
```

**优点**：
- 训练和推理特征分布一致
- 模型表现可预测
- 显存占用低

**缺点**：
- 推理时也无法利用高分辨率

#### 策略 2：训练和推理都用高分辨率

```yaml
# 训练时
max_dim: 1024

# 推理时
max_dim: 1024  # 保持一致
```

**优点**：
- 特征质量高
- 模型能学习到更精细的模式
- 推理效果好

**缺点**：
- 显存占用高
- 训练速度慢

#### 策略 3：根据任务需求选择

| 场景 | 推荐分辨率 | 理由 |
|------|-----------|------|
| 显存有限（<24GB） | 520-768 | 平衡质量和效率 |
| 显存充足（≥32GB） | 768-1024 | 更好的特征质量 |
| 需要识别小文字 | ≥768 | 保证可读性 |
| 主要是大元素操作 | 520-640 | 足够且高效 |
| 追求最佳效果 | 1024 | 最大化视觉特征质量 |

### 4.6 实际影响总结

| 特征类型 | 520分辨率 | 1024分辨率 | 影响程度 |
|---------|---------|-----------|---------|
| 全局布局 | ✓ 保留 | ✓ 保留 | 低 |
| 大元素（>100px） | ✓ 可识别 | ✓ 清晰 | 低 |
| 中元素（40-100px） | ⚠ 模糊 | ✓ 清晰 | 中 |
| 小元素（<40px） | ✗ 丢失 | ✓ 可识别 | 高 |
| 大文字（>20px） | ✓ 可读 | ✓ 清晰 | 低 |
| 小文字（<12px） | ✗ 不可读 | ✓ 可读 | 高 |
| 空间定位精度 | ±16px | ±4px | 高 |
| 细节特征 | ✗ 丢失 | ✓ 保留 | 高 |

### 4.7 关键原则

1. **训练和推理分辨率应保持一致**
   - 如果训练用 520，推理也用 520
   - 如果训练用 1024，推理也用 1024

2. **分辨率选择取决于任务需求**
   - 简单任务（大元素）：520 足够
   - 复杂任务（小文字、精确定位）：建议 ≥768

3. **只训练 attention 层仍然有效**
   - 只是学习的是"如何利用给定质量的视觉特征"
   - 特征质量由输入分辨率决定

4. **建议策略**
   - 先用 520 训练，如果效果可接受就继续
   - 如果需要更好效果，再提升到 768 或 1024
   - **但要确保训练和推理保持一致**

---

## 5. 如何微调视觉部分

### 5.1 查看模型层名称

在训练脚本中添加以下代码，打印所有模块名称：

```python
# 在 model 加载后添加
for name, module in model.named_modules():
    print(name)
```

或者只打印包含特定关键词的层：

```python
for name, module in model.named_modules():
    if any(k in name for k in ["visual", "vision", "patch", "proj"]):
        print(name)
```

### 5.2 扩展 target_modules 配置

根据模型实际结构，可以添加视觉编码器的模块。以下是示例配置：

```yaml
lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    # === 语言模型部分 ===
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
    
    # === 视觉编码器部分（需根据实际模型结构调整）===
    # Qwen-VL 系列常见的视觉层名称
    - visual.attn.qkv
    - visual.attn.proj
    - visual.mlp.fc1
    - visual.mlp.fc2
    
    # 或者使用通配符匹配（如果 PEFT 支持）
    # - "visual.*proj"
    # - "visual.*fc*"
```

### 5.3 注意事项

1. **显存增加**：添加视觉部分后显存占用会显著增加，可能需要：
   - 降低 batch size
   - 启用 4-bit 量化 (`use_4bit: true`)
   - 减小 LoRA rank (`r: 8`)

2. **学习率调整**：视觉和语言部分可能需要不同的学习率
   - 视觉部分通常使用更小的学习率（如 1e-5）
   - 语言部分可以用相对更大的学习率（如 2e-4）

3. **数据量要求**：微调视觉部分需要更多的训练数据，避免过拟合

---

## 6. 损失掩码机制说明

### 6.1 Multi-turn 对话格式中的损失计算

在 multi-turn 对话格式的训练数据中，**只有最后一个 assistant 响应参与损失计算**，所有历史的 assistant 响应都被掩盖（标记为 `-100`，不参与梯度更新）。

#### 6.1.1 训练数据格式示例

一个典型的 multi-turn 训练样本格式如下：

```
system message (系统提示)
user (instruction) ← 任务指令
assistant (step 1) ← 历史响应，不计算损失
assistant (step 2) ← 历史响应，不计算损失
...
assistant (step N-1) ← 历史响应，不计算损失
user (screenshot image) ← 当前截图
assistant (step N) ← **只有这个参与损失计算**
```

#### 6.1.2 损失掩码实现逻辑

训练代码中的 `_create_labels` 方法会：

1. **查找所有 assistant 响应位置**：通过识别 `<|im_start|>assistant` token 来定位
2. **只标记最后一个响应**：将最后一个 assistant 响应的 token 标记为有效标签（参与损失计算）
3. **掩盖其他所有内容**：
   - System message → `-100`（不计算损失）
   - User messages → `-100`（不计算损失）
   - 历史 assistant 响应 → `-100`（不计算损失）
   - 最后一个 assistant 响应 → 实际 token ID（计算损失）

#### 6.1.3 关键代码逻辑

```python
# 查找所有有效的 assistant token 位置
assistant_positions = []
for pos in range(L):
    if input_ids_flat[pos] == ASSISTANT_START_TOKEN:
        if pos > 0 and input_ids_flat[pos - 1] == IM_START_TOKEN:
            assistant_positions.append(pos)

# 只处理最后一个 assistant 响应
if assistant_positions:
    pos = assistant_positions[-1]  # 最后一个
    ans_start = pos + 2  # 跳过 "assistant" token 和换行符
    ans_end = ans_start
    while ans_end < L and input_ids_flat[ans_end] != IM_END_TOKEN:
        ans_end += 1
    if ans_end < L:
        # 只标记最后一个 assistant 响应的内容
        labels[ans_start : ans_end + 2] = input_ids[ans_start : ans_end + 2]
```

### 6.2 为什么只训练最后一个响应？

这种设计符合**"给定完整历史上下文，预测下一个正确动作"**的训练范式：

| 优点 | 说明 |
|------|------|
| **避免错误传播** | 历史中的错误动作不会干扰模型学习正确的下一步 |
| **聚焦当前任务** | 模型学习在给定完整历史的情况下，如何做出正确的下一步决策 |
| **符合推理场景** | 推理时也是基于历史上下文预测下一个动作，训练和推理一致 |
| **数据利用效率** | 每个训练样本专注于学习一个决策点，避免多目标冲突 |

### 6.3 训练效果

- ✅ **模型学习**：在给定完整历史（包括所有之前的动作和当前截图）的情况下，预测下一个正确的动作
- ✅ **上下文理解**：模型必须理解整个对话历史和当前状态，才能做出正确预测
- ✅ **避免过拟合**：不会学习到历史中可能存在的次优或错误动作序列

### 6.4 注意事项

1. **数据格式要求**：确保训练数据使用 multi-turn 格式，最后一个 assistant 响应是当前 step 要学习的动作
2. **历史长度**：虽然历史 assistant 响应不参与损失计算，但它们作为上下文输入，帮助模型理解当前状态
3. **参考实现**：这是参考 Qwen3-VL 官方训练代码的实现方式，符合标准的对话式训练范式

---

## 7. 推荐配置

### 7.1 仅微调 LLM（默认推荐）

适用于：指令遵循、格式调整、轻量级适配

```yaml
training:
  lora:
    enabled: true
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
  
  gradient_checkpointing: true   # 启用梯度检查点，显著降低显存占用。它让部分中间激活值不保留，仅在反向传播时临时重算，适合大模型微调。
  use_4bit: false
  
  sft:
    learning_rate: 2.0e-4
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 16
```

### 7.2 同时微调视觉和 LLM

适用于：领域差异大、需要学习新视觉概念

```yaml
training:
  lora:
    enabled: true
    r: 8  # 降低 rank 以节省显存
    alpha: 16
    dropout: 0.05
    target_modules:
      # LLM 部分
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
      # 视觉部分（根据实际模型调整）
      - visual.attn.qkv
      - visual.attn.proj
      - visual.mlp.fc1
      - visual.mlp.fc2
  
  gradient_checkpointing: true
  use_4bit: true  # 启用 4-bit 量化
  
  sft:
    learning_rate: 1.0e-4  # 适当降低学习率
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 32  # 增加累积步数
```

---

## 8. 常见问题

### Q1: 如何判断是否需要微调视觉部分？

**方法**：先用只调 LLM 的配置训练一版，评估效果：
- 如果模型能正确理解图像内容，但输出格式/行为不对 → 只调 LLM 足够
- 如果模型对特定视觉元素识别不准确 → 考虑加入视觉部分

### Q2: 添加视觉部分后 OOM 怎么办？

**解决方案**（按优先级）：
1. 启用 4-bit 量化：`use_4bit: true`
2. 减小 LoRA rank：`r: 8` 或 `r: 4`
3. 减小 batch size 并增加 gradient accumulation
4. 使用 DeepSpeed ZeRO 优化

### Q3: 如何验证 LoRA 作用在了哪些层？

在训练开始时，PEFT 会打印可训练参数信息：

```
trainable params: 12,345,678 || all params: 2,000,000,000 || trainable%: 0.62%
```

也可以手动检查：

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")
```

---

## 9. SFT 微调问题诊断与改进建议

### 8.1 小数据集 SFT 常见问题

当使用**小数据集**（<100 个样本）进行 SFT 微调时，常见问题包括：

| 问题 | 症状 | 原因 |
|------|------|------|
| **过拟合** | Loss 下降但推理效果差 | 数据量太少，模型记忆而非泛化 |
| **欠拟合** | Loss 不下降或波动大 | 学习率/训练步数不足，或数据质量差 |
| **泛化能力差** | 训练任务能完成，新任务失败 | 数据多样性不足，缺乏错误-纠正样本 |
| **执行错误后无法恢复** | 一步错后持续错误 | 缺乏错误恢复的训练数据 |

### 8.2 针对复杂任务场景的训练策略

#### 8.2.1 SFT vs RL 的选择

**建议：先用 SFT，再用 RL 精调**

| 阶段 | 方法 | 作用 | 数据需求 |
|------|------|------|---------|
| **阶段1：基础能力** | SFT | 学习任务执行模式、格式遵循 | 10-50条高质量轨迹 |
| **阶段2：鲁棒性** | SFT + 错误数据 | 学习错误恢复、边界情况处理 | 增加错误-纠正样本 |
| **阶段3：优化** | RL (GRPO/PPO) | 优化成功率、减少步数 | 在线交互数据 |

**为什么先 SFT？**
- ✅ SFT 能快速让模型学会基本执行模式
- ✅ 数据准备简单，不需要奖励函数
- ✅ 训练稳定，可复现
- ❌ 但无法学习错误恢复，容易过拟合

**什么时候需要 RL？**
- ✅ SFT 后模型能执行任务，但成功率不高（<70%）
- ✅ 需要优化执行效率（减少步数）
- ✅ 需要处理 SFT 数据中未覆盖的边缘情况

#### 8.2.2 数据增强策略

**1. 包含错误-纠正数据（强烈推荐）**

错误-纠正数据能显著提升模型的鲁棒性：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "打开移动云盘..."},
    {"role": "assistant", "content": "<thinking>错误：点击了错误的文件夹</thinking><tool_call>...</tool_call>"},
    {"role": "user", "content": "[错误后的截图]"},
    {"role": "assistant", "content": "<thinking>纠正：应该点击'销售部'文件夹</thinking><tool_call>...</tool_call>"}
  ]
}
```

**为什么需要错误数据？**
- ✅ 模型学习"识别错误状态"和"执行纠正动作"
- ✅ 提升执行错误后的恢复能力
- ✅ 减少"一步错，步步错"的问题

**如何构造错误数据？**
1. **人工构造**：在真实轨迹中插入1-2步错误动作，然后纠正
2. **数据增强**：对正确轨迹进行扰动（错误坐标、错误动作类型）
3. **失败轨迹**：收集实际执行失败的轨迹，标注纠正步骤

**2. 增加数据多样性**

| 策略 | 方法 | 效果 |
|------|------|------|
| **轨迹切片** | 将长轨迹切分为多个子任务 | 增加样本数，学习局部模式 |
| **变体生成** | 同一任务的不同执行路径 | 提升泛化能力 |
| **负样本** | 包含明显错误的样本 | 学习"什么不该做" |

**3. 数据质量检查**

确保每条数据包含：
- ✅ **清晰的思考过程**：说明为什么选择这个动作
- ✅ **准确的坐标**：坐标必须对应截图中的实际元素
- ✅ **完整的上下文**：包含足够的历史信息
- ✅ **正确的格式**：严格遵循 `<thinking>` 和 `<tool_call>` 格式

### 8.3 SFT 训练技巧

#### 8.3.1 小数据集训练配置优化

**当前配置问题分析：**

```yaml
# 当前配置（my_config_119.yaml）
num_train_epochs: 3          # ✅ 合理（小数据集需要多轮）
learning_rate: 1.0e-4        # ⚠️ 可能偏高，建议 5e-5 到 1e-4
per_device_train_batch_size: 1  # ✅ 合理
gradient_accumulation_steps: 4  # ⚠️ 有效batch size=4，可能太小
lora:
  r: 16                      # ⚠️ 小数据集建议 r=8-16
  alpha: 32                  # ✅ 2x r 合理
  target_modules:            # ⚠️ 只训练attention，可能不够
    - q_proj
    - k_proj
    - v_proj
    - o_proj
```

**改进建议配置：**

```yaml
training:
  sft:
    num_train_epochs: 5-10        # 小数据集需要更多轮次
    learning_rate: 5.0e-5         # 降低学习率，更稳定
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8  # 增大有效batch size
    warmup_steps: 5                # 小数据集需要warmup
    weight_decay: 0.01             # 添加权重衰减防过拟合
  
  lora:
    r: 8                          # 小数据集降低rank，减少过拟合风险
    alpha: 16
    dropout: 0.1                   # 增加dropout
    target_modules:                # 增加FFN层
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj                  # 添加FFN层
      - up_proj
      - down_proj
```

#### 8.3.2 训练监控与调试

**1. Loss 曲线分析**

| Loss 模式 | 含义 | 解决方案 |
|----------|------|---------|
| **持续下降** | ✅ 正常学习 | 继续训练 |
| **快速下降后平台** | ⚠️ 可能过拟合 | 检查验证集，早停 |
| **波动大** | ⚠️ 学习率过高或数据不一致 | 降低LR，检查数据质量 |
| **不下降** | ❌ 学习率太低或数据问题 | 提高LR，检查数据格式 |

**2. 添加验证集**

即使数据少，也要留出 20% 作为验证集：

```python
# 数据分割
train_size = int(0.8 * len(dataset))
train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:]

# 训练参数
training_args = TrainingArguments(
    eval_strategy="steps",
    eval_steps=10,  # 每10步评估一次
    save_strategy="steps",
    load_best_model_at_end=True,  # 加载最佳模型
    metric_for_best_model="eval_loss",
)
```

**3. 检查模型输出**

训练过程中定期检查模型生成：

```python
# 在训练脚本中添加
if step % 50 == 0:
    sample = next(iter(val_dataset))
    inputs = data_collator([sample])
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    print("Generated:", tokenizer.decode(outputs[0]))
```

#### 8.3.3 数据质量提升技巧

**1. 思考过程质量**

❌ **差的思考过程：**
```
<thinking>
Thought: I will click the button.
</thinking>
```

✅ **好的思考过程：**
```
<thinking>
Thought: I need to open the "销售部-部门共享" folder. 
Looking at the screen, I can see a folder icon with yellow color 
labeled "销售部" in the file list. I'll click on this folder 
to navigate into it.
</thinking>
```

**2. 坐标准确性**

- ✅ 使用**绝对坐标**（像素值），与截图尺寸对应
- ✅ 坐标必须指向**实际可点击的元素**
- ✅ 避免坐标落在**元素边界**或**重叠区域**

**3. 历史信息完整性**

确保每个样本包含：
- ✅ 完整的任务描述
- ✅ 所有之前的动作历史（至少前3-5步）
- ✅ 当前步骤的截图

### 8.4 针对你的具体问题的建议

#### 问题1：SFT 是否能学会复杂任务？

**答案：可以，但需要正确的数据策略**

- ✅ **SFT 能学会**：如果数据质量高、多样性足够
- ⚠️ **但需要**：
  - 10-50条高质量轨迹（你目前有10条，偏少）
  - 包含错误-纠正数据
  - 足够的训练轮次（5-10 epochs）
  - 合适的超参数配置

**建议：**
1. **先扩充数据到 30-50 条**（通过轨迹切片、变体生成）
2. **添加 5-10 条错误-纠正样本**
3. **训练 5-10 个 epochs**，监控验证集loss
4. **如果 SFT 后成功率 <70%**，再考虑 RL

#### 问题2：是否需要错误-纠正数据？

**答案：强烈建议添加**

**原因：**
- GUI Agent 任务中，**错误恢复能力**至关重要
- 你的数据只有正确轨迹，模型无法学习"错了怎么办"
- 实际执行时，一步错后容易"步步错"

**如何添加：**
1. **人工构造**：在10条轨迹中，选择3-5条，插入1-2步错误，然后纠正
2. **失败轨迹**：收集实际执行失败的案例，标注纠正步骤
3. **数据增强**：对正确动作进行扰动（坐标偏移、错误动作类型）

**示例：**
```json
// 错误-纠正样本
{
  "messages": [
    {"role": "user", "content": "打开移动云盘，找到销售部文件夹"},
    {"role": "assistant", "content": "<thinking>错误：点击了'技术部'文件夹</thinking><tool_call>...</tool_call>"},
    {"role": "user", "content": "[错误后的截图：显示技术部文件夹内容]"},
    {"role": "assistant", "content": "<thinking>纠正：应该点击'销售部'文件夹，我需要返回上一级</thinking><tool_call>{\"action\": \"system_button\", \"button\": \"back\"}</tool_call>"}
  ]
}
```

#### 问题3：SFT 技巧分享

**核心问题诊断：**

根据你的训练日志，Loss 在 0.15-0.4 之间波动，可能的原因：

1. **数据量太少**（31个样本）
   - 解决：扩充到 100+ 样本（轨迹切片、变体）

2. **学习率可能偏高**（1e-4）
   - 解决：降低到 5e-5，增加 warmup

3. **只训练 attention 层可能不够**
   - 解决：添加 FFN 层（gate_proj, up_proj, down_proj）

4. **缺乏错误-纠正数据**
   - 解决：添加 20-30% 的错误-纠正样本

**改进方案：**

```yaml
# 改进后的配置
training:
  sft:
    num_train_epochs: 8              # 增加轮次
    learning_rate: 5.0e-5           # 降低学习率
    gradient_accumulation_steps: 8   # 增大有效batch
    warmup_steps: 10                 # 添加warmup
    weight_decay: 0.01               # 防过拟合
  
  lora:
    r: 8                             # 降低rank
    alpha: 16
    dropout: 0.1                     # 增加dropout
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj                    # 添加FFN
      - up_proj
      - down_proj
```

**数据改进：**
1. **轨迹切片**：将10条长轨迹切分为 30-50 个子任务样本
2. **添加错误数据**：构造 10-15 条错误-纠正样本
3. **数据验证**：检查每条数据的坐标准确性、思考过程质量

**训练流程：**
1. **数据准备**：扩充到 100+ 样本，包含错误数据
2. **训练**：使用改进配置，训练 8-10 epochs
3. **验证**：每 10 步评估验证集，选择最佳checkpoint
4. **测试**：在真实任务上测试，如果成功率 <70%，考虑 RL

### 8.5 何时考虑 RL 训练

**考虑 RL 的时机：**
- ✅ SFT 后模型能执行任务，但成功率 <70%
- ✅ 需要优化执行效率（减少步数）
- ✅ 需要处理 SFT 数据中未覆盖的边缘情况
- ✅ 有在线交互环境可以收集数据

**RL 的优势：**
- ✅ 能学习错误恢复（通过奖励信号）
- ✅ 能优化长期策略（减少总步数）
- ✅ 能处理未见过的状态

**RL 的劣势：**
- ❌ 需要奖励函数设计
- ❌ 训练不稳定，需要调参
- ❌ 需要大量交互数据

**建议：先完善 SFT，再考虑 RL**

### 8.6 针对 189 个样本的参数调整建议

当你有 **189 个训练样本**（来自 11 条轨迹，每条轨迹拆分为多个 multi-turn 样本）时，参数调整如下：

#### 8.6.1 数据特点分析

| 特点 | 说明 | 影响 |
|------|------|------|
| **样本数量** | 189 个样本 | 中小型数据集，需要平衡过拟合和欠拟合 |
| **数据来源** | 11 条轨迹 | 数据多样性有限，需要正则化 |
| **样本结构** | Multi-turn 格式 | 每个样本包含历史上下文，需要足够的上下文长度 |
| **数据分布** | 同一轨迹的样本相关性强 | 需要打乱数据顺序，避免过拟合特定轨迹 |

#### 8.6.2 推荐配置（189 个样本）

```yaml
training:
  sft:
    max_length: 6200                    # 保持，multi-turn需要长上下文
    num_train_epochs: 5                  # 189样本：5个epochs足够（约945步）
    per_device_train_batch_size: 1      # 保持，VL模型显存限制
    gradient_accumulation_steps: 8       # 增大：有效batch size = 8
    learning_rate: 8.0e-5               # 189样本：适中学习率（比31样本高，比1000+样本低）
    warmup_steps: 19                    # 10% warmup: 189 * 5 / 8 / 10 ≈ 19步
    warmup_ratio: 0.1                   # 或使用warmup_ratio: 0.1
    logging_steps: 10                   # 每10步记录（189样本训练约95步/epoch）
    save_steps: 50                      # 每50步保存（约每0.5个epoch）
    save_total_limit: 3                  # 保留3个checkpoint
    eval_strategy: "steps"               # 添加验证集评估
    eval_steps: 50                      # 每50步评估验证集
    load_best_model_at_end: true       # 训练结束后加载最佳模型
    metric_for_best_model: "eval_loss"  # 使用验证集loss选择最佳模型
    weight_decay: 0.01                   # 添加权重衰减，防止过拟合
    bf16: false                          # V100不支持bf16
  
  # Memory optimization
  gradient_checkpointing: true
  use_4bit: true
  max_images_per_sample: 1              # 保持1张图片/样本
  
  # LoRA configuration
  lora:
    enabled: true
    r: 16                               # 189样本：保持r=16（数据量适中）
    alpha: 32                           # 2x r，保持
    dropout: 0.1                        # 增加dropout到0.1，防止过拟合
    target_modules:                     # 添加FFN层，提升表达能力
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj                       # 添加FFN层
      - up_proj
      - down_proj
```

#### 8.6.3 关键参数说明

**1. 训练轮次（num_train_epochs: 5）**

- **计算**：189 样本 × 5 epochs = 945 训练步（batch_size=1, gradient_accumulation=8）
- **理由**：
  - 189 个样本不算太少，5 个 epochs 足够学习
  - 过多 epochs（>8）容易过拟合 11 条轨迹
  - 可以通过验证集 loss 早停

**2. 学习率（learning_rate: 8.0e-5）**

- **对比**：
  - 31 样本：5e-5（更保守）
  - 189 样本：8e-5（适中）
  - 1000+ 样本：1e-4 到 2e-4（可以更高）
- **理由**：189 个样本数据量适中，学习率可以比极小数据集稍高

**3. 有效 Batch Size（gradient_accumulation_steps: 8）**

- **计算**：1 × 8 = 8 个样本/更新
- **理由**：
  - 增大有效 batch size 提升训练稳定性
  - 189 个样本可以支持更大的有效 batch
  - 每个 epoch 约 24 次更新（189/8 ≈ 24）

**4. Warmup Steps（warmup_steps: 19）**

- **计算**：总步数 = 189 × 5 / 8 ≈ 118 步，10% warmup ≈ 19 步
- **理由**：小数据集需要 warmup 稳定训练初期

**5. LoRA Rank（r: 16）**

- **对比**：
  - 31 样本：r=8（降低过拟合风险）
  - 189 样本：r=16（平衡表达能力和过拟合）
  - 1000+ 样本：r=32-64（可以更高）
- **理由**：189 个样本可以支持 r=16，同时通过 dropout=0.1 防止过拟合

**6. 添加 FFN 层**

- **原因**：189 个样本数据量足够，添加 FFN 层（gate_proj, up_proj, down_proj）可以提升模型表达能力
- **显存影响**：FFN 层会增加约 30-40% 的可训练参数，但 4-bit 量化下仍可接受

#### 8.6.4 数据分割建议

```python
# 建议的数据分割（在训练脚本中）
from sklearn.model_selection import train_test_split

# 按轨迹分割，避免同一轨迹的样本同时出现在训练集和验证集
trajectory_ids = [sample['metadata']['source'] for sample in all_samples]
unique_trajectories = list(set(trajectory_ids))

# 80% 轨迹用于训练，20% 用于验证
train_trajs, val_trajs = train_test_split(
    unique_trajectories, 
    test_size=0.2, 
    random_state=42
)

train_samples = [s for s in all_samples if s['metadata']['source'] in train_trajs]
val_samples = [s for s in all_samples if s['metadata']['source'] in val_trajs]

print(f"训练集: {len(train_samples)} 样本")
print(f"验证集: {len(val_samples)} 样本")
```

**为什么按轨迹分割？**
- ✅ 避免数据泄露：同一轨迹的样本相关性高
- ✅ 更真实的验证：验证集包含完整的新轨迹
- ✅ 更好的泛化评估：测试模型在未见过的轨迹上的表现

#### 8.6.5 训练监控指标

**关键指标：**

1. **训练 Loss**：应该持续下降，最终稳定在 0.1-0.2 左右
2. **验证 Loss**：应该跟随训练 loss 下降，如果验证 loss 开始上升而训练 loss 继续下降 → 过拟合
3. **验证 Loss 最低点**：选择验证 loss 最低的 checkpoint 作为最终模型

**预期训练曲线：**
```
Epoch 1: train_loss ~0.4-0.5, eval_loss ~0.5-0.6
Epoch 2: train_loss ~0.2-0.3, eval_loss ~0.3-0.4
Epoch 3: train_loss ~0.15-0.25, eval_loss ~0.2-0.3
Epoch 4: train_loss ~0.1-0.2, eval_loss ~0.15-0.25
Epoch 5: train_loss ~0.1-0.15, eval_loss ~0.15-0.2
```

#### 8.6.6 与当前配置的对比

| 参数 | 当前配置（31样本） | 推荐配置（189样本） | 变化原因 |
|------|-------------------|-------------------|---------|
| `num_train_epochs` | 3 | 5 | 数据量增加，需要更多轮次学习 |
| `learning_rate` | 1.0e-4 | 8.0e-5 | 数据量适中，学习率适中 |
| `gradient_accumulation_steps` | 4 | 8 | 数据量增加，可以增大有效batch |
| `warmup_steps` | 1 | 19 | 需要足够的warmup稳定训练 |
| `lora.r` | 16 | 16 | 保持，数据量适中 |
| `lora.dropout` | 0.05 | 0.1 | 增加dropout防止过拟合 |
| `lora.target_modules` | 只有attention | + FFN层 | 数据量足够，添加FFN提升能力 |
| `eval_strategy` | 无 | "steps" | 添加验证集评估，选择最佳模型 |

#### 8.6.7 关于 4-bit 量化的影响

**4-bit 量化对训练效果的影响：**

| 方面 | 影响 | 说明 |
|------|------|------|
| **显存占用** | ✅ 大幅减少 | 约减少 75% 显存，从 ~24GB → ~6GB |
| **训练速度** | ✅ 可能更快 | 更小的模型 → 更快的计算 |
| **模型精度** | ⚠️ 有损失 | 权重从 FP16/BF16 → 4-bit，量化误差 |
| **训练稳定性** | ⚠️ 可能不稳定 | 量化误差影响梯度计算准确性 |
| **小数据集影响** | ❌ **影响更明显** | 量化误差在小数据集上更突出 |

**为什么小数据集受影响更大？**

1. **量化误差累积**：小数据集训练步数少，量化误差难以通过大量更新被"平均掉"
2. **梯度精度**：4-bit 量化的梯度计算精度较低，小数据集对梯度精度更敏感
3. **表达能力受限**：量化后的模型表达能力下降，小数据集更需要模型的完整表达能力

**建议：**

**方案1：如果显存足够（推荐）**

```yaml
training:
  use_4bit: false  # 关闭4-bit量化，使用FP16/BF16
  gradient_checkpointing: true  # 仍然需要，节省显存
```

**显存估算（关闭4-bit）：**
- MAI-UI-2B 模型：~4GB（FP16）
- LoRA 参数：~50MB
- 训练时激活值：~8-12GB（取决于 batch size 和序列长度）
- **总计：~12-16GB**（V100 32GB 足够）

**方案2：如果显存不足（必须使用4-bit）**

```yaml
training:
  use_4bit: true  # 必须使用4-bit
  # 通过其他方式补偿量化损失：
  lora:
    r: 32  # 提高LoRA rank，补偿量化损失
    alpha: 64
  sft:
    learning_rate: 1.0e-4  # 稍微提高学习率
    num_train_epochs: 6  # 增加训练轮次
```

**对比实验建议：**

1. **先尝试关闭4-bit训练**（如果显存允许）
   - 如果显存占用 < 24GB，强烈建议关闭4-bit
   - 预期效果提升：5-15%

2. **如果必须使用4-bit**
   - 提高 LoRA rank（r=16 → r=32）
   - 增加训练轮次（5 → 6-8 epochs）
   - 稍微提高学习率（8e-5 → 1e-4）

**如何判断显存是否足够？**

```python
# 在训练脚本中添加显存监控
import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载模型后检查
    model_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"Model memory: {model_memory:.1f} GB")
    
    # 如果模型内存 < 16GB，可以尝试关闭4-bit
    if model_memory < 16:
        print("建议：可以尝试关闭4-bit量化以获得更好的训练效果")
```

#### 8.6.8 如果效果仍不理想

如果按照上述配置训练后效果仍不理想，考虑：

1. **数据质量检查**
   - 检查坐标准确性（坐标是否指向正确元素）
   - 检查思考过程质量（是否清晰说明为什么选择这个动作）
   - 检查历史信息完整性（是否包含足够的上下文）

2. **添加错误-纠正数据**
   - 在 189 个样本基础上，添加 30-50 个错误-纠正样本
   - 提升模型的错误恢复能力

3. **调整超参数**
   - 如果过拟合：降低 r=8，增加 dropout=0.15，增加 weight_decay=0.05
   - 如果欠拟合：提高 r=32，降低 dropout=0.05，增加训练轮次到 8

4. **考虑 RL 训练**
   - 如果 SFT 后成功率 <70%，考虑使用 RL（GRPO/PPO）进一步优化

---

## 10. 参考资料

- [PEFT 官方文档](https://huggingface.co/docs/peft)
- [Qwen-VL 技术报告](https://arxiv.org/abs/2308.12966)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
