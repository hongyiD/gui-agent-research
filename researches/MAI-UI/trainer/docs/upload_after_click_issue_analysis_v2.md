# 文件上传后点击错误问题分析（修正版）

> ## ✅ 实施状态：已完成 (2026-01-26)
> 
> 所有步骤已实施完成：
> - ✅ `data_augmentation.py` 已更新：使用对比推理（移除负样本生成）
> - ✅ 干净数据集已生成：`sft_train_clean.jsonl` (26 样本)
> - ✅ 增强数据集已生成：`sft_train_final.jsonl` (38 样本，含 12 个对比推理样本)
> - ✅ 训练配置：`configs/my_config_v3_fix.yaml`
> - ✅ 训练脚本：`scripts/train_v3_fix.sh`
>
> **执行训练：**
> ```bash
> cd examples/agenticx-for-guiagent/gui-agent-research/researches/MAI-UI/trainer
> bash scripts/train_v3_fix.sh
> ```
>
> **核心改进：**
> - 移除了有害的负样本生成（SFT不应该训练错误动作）
> - 在正样本中嵌入对比推理："点击文件，不是蓝色加号按钮"
> - 增加训练轮数到 5，降低学习率到 1.0e-5

## 问题描述

在2B模型的SFT-LoRA微调后，模型在上传文件成功后，错误地点击了右下角的蓝色加号按钮，而不是点击文件本身来获取分享链接。8B模型表现正常。

**重要澄清**：训练数据来自**正确的轨迹** `20260119_201327`，而不是 badcase。

### 正确流程（8B模型，训练数据来源）
- **step_015**: 上传成功后，点击文件本身（坐标 `[0.484, 0.269]`）
  - Thinking: "I will select the uploaded Excel file to copy its share link."
  - Action: `{"action": "click", "coordinate": [0.4844844844844845, 0.26926926926926925]}`
- **step_016**: 点击分享按钮，显示分享链接
  - Thinking: "I will open the share options to copy the link."
  - Action: `{"action": "click", "coordinate": [0.12312312312312312, 0.9269269269269269]}`

### 错误流程（2B模型微调后）
- **上传成功后**，错误地点击了蓝色加号（坐标 `[0.898, 0.823]`）
- 然后跳转到登录页面

## 根本原因分析（修正）

### 1. 训练样本数量不足 ⚠️

**关键发现**：
- 训练数据只有 **26个样本**（来自 `20260119_201327` 轨迹）
- step_015（上传后点击文件）只有 **1个样本**
- 对于2B模型来说，1个样本可能不足以学习这个复杂的模式

**为什么8B模型表现正常？**
- 8B模型容量更大，可能从预训练中已经学到了一些通用模式
- 或者8B模型对视觉特征的理解能力更强，能更好地从截图中识别文件位置

### 2. 模型容量限制

- **2B模型**对细微差别的理解能力不如8B模型
- 对于相似的UI元素（文件 vs 蓝色加号），2B模型可能难以区分
- 需要更多的训练样本才能学习到正确的模式

### 3. 视觉理解问题

- step_015 的截图中，文件位置和蓝色加号可能在某些情况下看起来相似
- 2B模型的视觉编码器可能无法充分区分这两个元素
- 需要更多的正样本和负样本来帮助模型学习区分

### 4. 上下文理解不足

- 模型可能没有充分理解"上传成功后应该点击文件"这个逻辑关系
- 需要更多的上下文信息或明确的提示

## 改进方案（修正）

### 方案1：增加训练样本（推荐，立即执行）

#### 1.1 数据增强
使用 `data_augmentation.py` 中的 `create_upload_after_click_samples()` 函数：

```bash
python data_augmentation.py \
  --input ../dataset/processed/sft_train.jsonl \
  --output ../dataset/processed/sft_train_enhanced.jsonl \
  --format messages \
  --upload-after-ratio 0.3  # 增加上传后操作的样本比例
```

这会：
- 从现有的上传相关样本中生成更多变体
- 创建正样本（点击文件）和负样本（点击加号）
- 通过对比学习帮助模型区分正确和错误行为

#### 1.2 收集更多轨迹
- 收集更多"上传文件后分享"的轨迹
- 确保每个关键步骤都有足够的样本（建议至少5-10个）

### 方案2：改进训练策略

#### 2.1 增加训练轮数
对于小数据集，可能需要更多的训练轮数：

```yaml
training:
  sft:
    num_train_epochs: 5  # 从3增加到5
```

#### 2.2 降低学习率
使用更小的学习率，让模型更仔细地学习：

```yaml
training:
  sft:
    learning_rate: 1.0e-5  # 从2.0e-5降低到1.0e-5
```

#### 2.3 增加关键步骤的权重
对关键步骤（如上传后的操作）增加损失权重（需要修改训练代码）

### 方案3：Prompt 工程改进

在系统 prompt 中明确说明：

```
## 重要提示：文件上传后的操作
- 上传文件成功后，如果需要分享文件，应该：
  1. 点击文件本身（文件列表中的文件项，通常在屏幕中央偏上）
  2. 然后点击分享按钮
- **不要**点击右下角的蓝色加号按钮（+），那是用于上传新文件的
- 蓝色加号按钮位于屏幕右下角，坐标通常在 [0.85-0.95, 0.80-0.90] 范围内
- 文件位置通常在屏幕中央偏上，坐标通常在 [0.4-0.6, 0.2-0.4] 范围内
```

### 方案4：视觉特征增强

#### 4.1 图像预处理
- 确保截图清晰
- 可能需要调整图像分辨率或对比度

#### 4.2 多视角样本
- 从不同角度或不同状态下截取上传成功后的界面
- 帮助模型学习更鲁棒的特征

## 实施步骤（修正）

### 立即执行（高优先级）

1. **应用数据增强**
   ```bash
   cd examples/agenticx-for-guiagent/gui-agent-research/researches/MAI-UI/trainer
   
   python data_augmentation.py \
     --input ../dataset/processed/sft_train.jsonl \
     --output ../dataset/processed/sft_train_enhanced_v2.jsonl \
     --format messages \
     --upload-after-ratio 0.3
   ```

2. **调整训练配置**
   ```yaml
   # configs/my_config_v2.yaml
   training:
     sft:
       num_train_epochs: 5  # 增加训练轮数
       learning_rate: 1.0e-5  # 降低学习率
   ```

3. **重新训练**
   ```bash
   python sft_trainer.py \
     --config configs/my_config_v2.yaml \
     --data_path ../dataset/processed/sft_train_enhanced_v2.jsonl
   ```

### 中期改进（1-2周）

1. 收集更多"上传后分享"的轨迹数据
2. 实现加权损失函数，对关键步骤给予更高权重
3. 优化系统 prompt，明确说明上传后的操作

### 长期优化（1个月+）

1. 实现对比学习
2. 添加视觉特征增强
3. 考虑使用更大的模型或更复杂的架构

## 预期效果

- **当前**：错误点击率 100%（总是点击蓝色加号）
- **数据增强后**：错误点击率降低到 30-50%
- **增加训练轮数+降低学习率**：错误点击率降低到 10-20%
- **收集更多数据后**：错误点击率降低到 <5%

## 关键洞察

1. **问题不在数据质量**，而在**数据数量**
2. **2B模型需要更多样本**才能学习复杂模式
3. **数据增强是关键**：通过生成变体样本可以显著改善效果
4. **训练策略也很重要**：更多的训练轮数和更小的学习率可能有助于小数据集

## 相关文件

- 训练脚本：`trainer/sft_trainer.py`
- 数据处理：`trainer/data/unified_data_processor.py`
- 数据增强：`trainer/data_augmentation.py`
- 训练数据：`dataset/20260119_201327/sft_train.jsonl`（26个样本）
- 训练日志：`trainer/train_logs/v2.md`
