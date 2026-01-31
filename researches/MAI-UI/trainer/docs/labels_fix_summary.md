# Labels 创建方法修复总结

## 问题描述

`sft_trainer.py` 中的 `MultiModalDataCollator._create_labels` 方法使用了粗糙的启发式方法来创建训练 labels，导致以下问题：

1. **不准确**：单独 tokenize response 文本，假设最后 N 个 token 是 response
2. **忽略特殊 token**：没有考虑 `<|im_start|>` 和 `<|im_end|>` 的位置
3. **忽略图像 token**：没有考虑图像 token 的影响
4. **可能导致错误的 loss 计算**：prompt 的一部分可能被错误计算 loss，或 response 的一部分被忽略

## 修复方案

采用官方 qwenvl 实现的方法，基于 token ID 精确定位 assistant response：

### 关键改进

1. **动态获取 token ID**：
   - 通过 `tokenizer.encode("<|im_start|>assistant")` 动态获取 token ID
   - 从 `tokenizer.eos_token_id` 获取 `<|im_end|>` token ID
   - 如果动态获取失败，回退到已知值（77091, 151645）

2. **精确定位**：
   - 在完整的 tokenized 序列中查找 `<|im_start|>assistant` token
   - 查找对应的 `<|im_end|>` token
   - 只对 assistant response 部分（ans_start 到 ans_end+1）设置 labels

3. **兼容性**：
   - 动态检查换行符，兼容不同的格式
   - Fallback 机制确保在不同模型版本下都能工作

## 代码变更

### 修改的文件

- `examples/agenticx-for-guiagent/gui-agent-research/researches/MAI-UI/trainer/sft_trainer.py`
  - 修改 `MultiModalDataCollator._create_labels` 方法（1174-1245行）

### 新增的文件

- `examples/agenticx-for-guiagent/gui-agent-research/researches/MAI-UI/trainer/scripts/verify_token_ids.py`
  - 验证 token ID 获取脚本
  
- `examples/agenticx-for-guiagent/gui-agent-research/researches/MAI-UI/trainer/scripts/test_labels_creation.py`
  - 测试 labels 创建逻辑脚本

## 验证信息

从 `maiui_2b_modelfiles/config.json` 确认：
- `eos_token_id: 151645` ✅（对应 `<|im_end|>`）
- `model_type: "qwen3_vl"` ✅

从 `chat_template.jinja` 确认对话格式：
- `<|im_start|>assistant\n` 开始 assistant 回复
- `<|im_end|>\n` 结束

## 使用方法

### 验证 token ID

```bash
cd examples/agenticx-for-guiagent/gui-agent-research/researches/MAI-UI/trainer
python scripts/verify_token_ids.py --model_path Tongyi-MAI/MAI-UI-2B
```

### 测试 labels 创建

```bash
cd examples/agenticx-for-guiagent/gui-agent-research/researches/MAI-UI/trainer
python scripts/test_labels_creation.py
```

## 参考实现

官方实现参考：
- `examples/agenticx-for-guiagent/gui-agent-research/researches/MAI-UI/Qwen3-VL/qwen-vl-finetune/qwenvl/data/data_processor.py:202-241`

## 后续优化（可选）

1. **添加 position_ids 支持**：
   - 导入 `get_rope_index_3` 从 `qwenvl.data.rope2d`
   - 在 collator 中计算 position_ids（需要图像 grid_thw 信息）

2. **优化数据预处理流程**：
   - 考虑在 `PreprocessedMultiModalDataset` 中预处理时使用 `apply_chat_template`
   - 减少在 collator 中的处理开销
