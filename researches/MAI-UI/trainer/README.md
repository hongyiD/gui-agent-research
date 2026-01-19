# MAI-UI SFT + RL 训练代码（企业级工程化版本）

**作者**: Damon Li  
**日期**: 2026年1月19日  
**版本**: v2.0 - 企业级工程化版本

本项目基于 MAI-UI 技术报告，实现了一套**生产就绪**的端到端 SFT + RL 训练流程，完全对齐上游 MobileWorld 代码库，并进行了企业级工程化改造。

## 核心特性

### 🎯 基础能力
- **上游对齐**: 完全对齐上游 `mobile_world` 的动作格式、轨迹日志格式和 Prompt 模板
- **自进化数据管道**: 从 `traj.json` 日志构建 SFT 数据集，支持拒绝采样、人工标注、自动 Agent 执行三种数据源
- **基于 verl 的异步 RL**: 实现严格 on-policy 异步 RL 训练框架，支持大规模并行环境
- **经验回放**: 实现论文中的经验回放机制（仅在无成功轨迹时补充）

### 🚀 工程化增强（v2.0 新增）
- **统一数据处理器**: 支持多种输出格式（OpenAI Messages、Prompt-Response、Full Trajectory）
- **端到端 Pipeline 编排**: 自动串联数据处理、训练、评估全流程，支持断点续传
- **批量评估系统**: 自动测试所有 checkpoint，生成性能对比报告
- **配置驱动**: 统一配置文件 `unified_config.yaml`，支持 CLI 参数覆盖
- **数据验证**: 自动验证数据格式、路径合法性、参数完整性
- **一键启动**: `quick_start.sh` 脚本，快速启动完整训练流程

## 目录结构

```
trainer/
├── configs/                          # 配置文件目录
│   ├── unified_config.yaml          # 统一配置入口（推荐使用）⭐
│   ├── data_config.yaml             # 数据构建配置（独立使用）
│   ├── sft_config.yaml              # SFT 训练配置（独立使用）
│   └── rl_config.yaml               # RL 训练配置（独立使用）
├── data/                            # 数据处理模块
│   ├── unified_data_processor.py    # 统一数据处理器（推荐）⭐
│   ├── build_data.py                # 原始数据构建脚本
│   ├── data_formats.py              # 标准数据格式定义
│   ├── format_validators.py         # 数据格式验证器
│   └── process_trajectory_jsonl.py  # 轨迹 JSONL 处理脚本
├── pipeline/                        # Pipeline 编排模块 ⭐
│   ├── orchestrator.py              # 端到端流程编排器
│   ├── checkpoint_manager.py        # 检查点管理器
│   └── pipeline_config.yaml         # Pipeline 配置文件
├── evaluation/                      # 批量评估模块 ⭐
│   ├── batch_evaluator.py           # 批量评估器
│   ├── metrics_analyzer.py          # 指标分析器
│   └── report_generator.py          # 报告生成器
├── scripts/                         # 工具脚本 ⭐
│   ├── quick_start.sh               # 一键启动完整流程
│   ├── validate_data.py             # 数据验证工具
│   ├── resume_training.py           # 恢复训练工具
│   └── compress_images.py           # 图片压缩工具
├── docs/                            # 详细文档 ⭐
│   ├── DATA_FORMAT.md               # 数据格式详细说明
│   └── TROUBLESHOOTING.md           # 常见问题排查指南
├── mobile_world/                    # 上游 MobileWorld 代码库
├── sft_trainer.py                   # SFT 训练脚本
├── rl_trainer.py                    # RL 训练脚本（verl 风格）
├── evaluate.py                      # 模型评估脚本（使用上游 runner）
├── requirements.txt                 # 依赖列表
└── README.md                        # 本文档

⭐ 标记为 v2.0 新增的企业级功能
```

## 与上游代码对齐

### 动作格式

所有动作使用上游的 `JSONAction` 模型，支持以下动作类型：
- GUI 操作: `click`, `long_press`, `double_tap`, `drag`, `swipe`, `input_text`, `navigate_back`, `navigate_home`, `open_app`, `wait`
- 特殊操作: `ask_user`, `answer`, `finished`, `mcp`

### 轨迹格式

轨迹数据完全对齐上游 `TrajLogger` 格式：
- `traj.json` 包含 `task_goal`, `step`, `prediction`, `action`, `ask_user_response`, `tool_call`
- 截图存储在 `screenshots/` 目录
- 评分存储在 `result.txt`

### Prompt 格式

SFT 数据构建使用上游 `MAI_MOBILE_SYS_PROMPT_ASK_USER_MCP` 格式，包含：
- System prompt（动作空间定义）
- Task goal
- Action history（最近 3 步）
- Current observation（截图）

---

## 快速开始（推荐新用户）⭐

### 方式 1：一键启动（最简单）

```bash
# 1. 准备配置文件
cp configs/unified_config.yaml configs/my_config.yaml

# 2. 编辑配置文件，设置以下必需参数：
#    - data.log_root: 原始轨迹日志根目录
#    - sft.model_path: 预训练模型路径
#    - rl.llm_base_url: LLM API 基础 URL

# 3. 一键启动完整流程（数据处理 → SFT → RL → 评估）
bash scripts/quick_start.sh --config configs/my_config.yaml
```

脚本会自动完成：
- ✅ 数据验证和预处理
- ✅ SFT 训练
- ✅ RL 训练
- ✅ 批量评估所有 checkpoint
- ✅ 生成性能对比报告

### 方式 2：使用 Pipeline 编排器（推荐高级用户）

```bash
# 使用 Python API 进行更灵活的控制
python -m pipeline.orchestrator \
    --config configs/unified_config.yaml \
    --stages data,sft,rl,eval \
    --resume_from_checkpoint  # 支持断点续传
```

Pipeline 编排器特性：
- 🔄 **断点续传**: 训练中断后自动从上次位置恢复
- 📊 **实时监控**: 自动记录实验日志和 metrics
- 🎯 **灵活配置**: 可选择执行特定阶段
- 🛡️ **错误恢复**: 自动处理常见错误并重试

### 方式 3：分步执行（完全控制）

参见下方"详细使用流程"章节。

---

## 详细使用流程

### 1. 环境准备

```bash
cd trainer
pip install -r requirements.txt
```

确保上游代码在 `../upstream/src` 目录下可用。

### 2. 数据构建

#### 选项 A：统一数据处理器（推荐）⭐

支持多种输出格式，自动验证数据质量：

```bash
# 使用统一数据处理器
python data/unified_data_processor.py \
    --config configs/data_config.yaml \
    --output_format prompt_response \  # 可选: openai_messages, full_trajectory
    --validate  # 自动验证输出数据
```

**支持的输出格式**：
- `openai_messages`: OpenAI API 兼容格式（用于 API 部署）
- `prompt_response`: Prompt-Response 格式（用于 SFT 训练，推荐）
- `full_trajectory`: 完整轨迹格式（用于 RL 训练）

**数据验证功能**：
- ✅ 自动检查图片路径是否存在
- ✅ 验证动作格式是否合法
- ✅ 统计样本数、动作类型分布、平均步数
- ✅ 生成数据质量报告

详细格式说明请参考 [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)

#### 选项 B：原始数据构建脚本

从上游运行产生的 `traj.json` 日志构建 SFT 数据集：

```bash
# 编辑 configs/data_config.yaml 设置 log_root 路径
python data/build_data.py --config configs/data_config.yaml
```

输出为 JSONL 格式，每行包含：
- `prompt`: 完整的 prompt（包含 system prompt、task goal、history、image）
- `response`: 模型的原始 prediction
- `metadata`: 任务名称、步骤、动作等元数据

### 3. SFT 训练

使用构建的数据进行监督微调：

```bash
# 编辑 configs/sft_config.yaml 设置模型和数据路径
python sft_trainer.py --config configs/sft_config.yaml

# 或使用 CLI 参数覆盖
python sft_trainer.py --config configs/sft_config.yaml \
    --model_name_or_path Tongyi-MAI/MAI-UI-2B \
    --data_path ./data/processed/sft_data.jsonl \
    --output_dir ./models/sft_model
```

### 4. RL 训练（基于 verl 的异步 on-policy）

使用上游环境进行异步 RL 训练：

```bash
# 编辑 configs/rl_config.yaml 设置所有必需参数
python rl_trainer.py --config configs/rl_config.yaml --llm_base_url https://api.openai.com/v1

# 必需配置项：
# - model.sft_model_path: SFT 模型路径
# - llm_base_url: LLM API 基础 URL
# - environment.num_parallel_envs: 并行环境数量
```

**RL 训练特性**：
- **异步 Rollout**: 使用线程池管理环境池，异步执行 rollout
- **严格 On-Policy**: 每轮更新仅使用当前策略的 rollout 轨迹
- **经验回放**: 维护成功轨迹缓冲，仅在无成功轨迹时补充
- **奖励设计**: 任务完成奖励 + 动作重复惩罚
- **DAPO 风格**: 支持非对称裁剪（当前使用对称裁剪，可扩展）

**混合并行支持**：
- 当前实现预留了混合并行接口（`hybrid_parallelism` 配置项）
- 实际 TP/PP/CP 需要集成 Megatron-LM 或其他框架
- 长序列训练可通过调整 `max_step` 和 batch 大小适配

### 5. 模型评估

#### 选项 A：批量评估系统（推荐）⭐

自动测试所有 checkpoint，生成性能对比报告：

```bash
# 批量评估所有 checkpoint
python -m evaluation.batch_evaluator \
    --model_dir ./models/sft_model \  # 模型目录
    --checkpoint_pattern "checkpoint-*" \  # checkpoint 匹配模式
    --eval_config configs/eval_config.yaml \
    --output_dir ./eval_results

# 生成性能对比报告
python -m evaluation.report_generator \
    --eval_results_dir ./eval_results \
    --output_format markdown  # 可选: html, json
```

**批量评估特性**：
- 🔄 自动发现所有 checkpoint
- 📊 并行评估，支持多 GPU
- 📈 生成 Markdown/HTML 性能对比报告
- 📉 可视化学习曲线（任务成功率、平均步数等）

**评估报告示例**：

| Checkpoint | 任务成功率 | 平均步数 | 动作重复率 | 评估时间 |
|-----------|----------|---------|----------|---------|
| step-100  | 45.2%    | 18.3    | 12.5%    | 2.3h    |
| step-200  | 62.8%    | 15.7    | 8.3%     | 2.5h    |
| final     | 78.5%    | 13.2    | 5.1%     | 2.7h    |

#### 选项 B：单模型评估

使用上游 runner 进行单个模型评测：

```bash
python evaluate.py \
    --agent_type mai_ui_agent \
    --model_name Tongyi-MAI/MAI-UI-2B \
    --llm_base_url https://api.openai.com/v1 \
    --log_root ./eval_logs \
    --tasks task1,task2,task3 \
    --max_step 50 \
    --enable_mcp  # 如果任务需要 MCP 工具
```

评估脚本完全复用上游 `run_agent_with_evaluation`，确保：
- 动作解析与训练时一致
- 轨迹日志格式统一
- 评分逻辑一致

---

## 工具脚本（v2.0 新增）⭐

### quick_start.sh - 一键启动完整流程

```bash
# 基础用法
bash scripts/quick_start.sh --config configs/unified_config.yaml

# 高级选项
bash scripts/quick_start.sh \
    --config configs/my_config.yaml \
    --skip_data_processing \  # 跳过数据处理
    --skip_sft \               # 跳过 SFT 训练
    --only_eval                # 仅执行评估
```

### validate_data.py - 数据验证工具

在训练前验证数据质量：

```bash
# 验证数据集
python scripts/validate_data.py \
    --data_path ./data/processed/sft_data.jsonl \
    --output_report ./validation_report.json

# 验证结果示例：
# ✅ 数据格式: 100% 合法
# ✅ 图片路径: 98.5% 存在（3 个缺失）
# ✅ 动作格式: 100% 合法
# 📊 样本总数: 1523
# 📊 平均步数: 12.3
# 📊 动作分布: click(45%), input_text(23%), swipe(18%), ...
```

### resume_training.py - 恢复训练工具

从中断点恢复训练：

```bash
# 自动检测最新 checkpoint 并恢复
python scripts/resume_training.py \
    --model_dir ./models/sft_model \
    --config configs/sft_config.yaml

# 指定 checkpoint 恢复
python scripts/resume_training.py \
    --checkpoint_path ./models/sft_model/checkpoint-1000 \
    --config configs/sft_config.yaml
```

### compress_images.py - 图片压缩工具

减小数据集体积，加速训练：

```bash
# 批量压缩图片
python scripts/compress_images.py \
    --input_dir ./data/raw/screenshots \
    --output_dir ./data/processed/screenshots \
    --quality 85 \  # JPEG 质量（1-100）
    --max_size 1024  # 最大边长（像素）

# 可减小 50-70% 体积，同时保持训练质量
```

---

## 配置说明

### unified_config.yaml - 统一配置文件（推荐）⭐

一个配置文件管理所有流程：

```yaml
# 数据处理配置
data:
  log_root: /path/to/trajectory/logs
  output_format: prompt_response
  image_processing:
    compress: true
    quality: 85

# SFT 训练配置
sft:
  model_path: Tongyi-MAI/MAI-UI-2B
  data_path: ./data/processed/sft_data.jsonl
  output_dir: ./models/sft_model
  training:
    batch_size: 4
    learning_rate: 2e-5
    num_epochs: 3

# RL 训练配置
rl:
  sft_model_path: ./models/sft_model
  llm_base_url: https://api.openai.com/v1
  num_parallel_envs: 8
  ppo:
    clip_epsilon: 0.2
    learning_rate: 1e-5

# 评估配置
evaluation:
  tasks: task1,task2,task3
  max_step: 50
  enable_mcp: true
```

### data_config.yaml

- `data_sources.log_root`: 上游日志根目录
- `filtering.min_score`: 最小成功分数阈值
- `image_processing.image_format`: 图片格式（`path` 或 `base64`）
- `output.output_path`: 输出 JSONL 文件路径

### sft_config.yaml

- `model.path`: 预训练模型路径
- `data.path`: SFT 数据 JSONL 路径
- `training.*`: 训练超参数（batch size、learning rate 等）

### rl_config.yaml

- `model.sft_model_path`: SFT 模型路径
- `llm_base_url`: LLM API 基础 URL（必需）
- `environment.num_parallel_envs`: 并行环境数量
- `ppo.*`: PPO 超参数
- `replay_buffer_size`: 经验回放缓冲大小（每任务最多轨迹数）

## 关键实现要点（与 verl 对齐）

### 严格 On-Policy

- 每轮更新仅使用当前策略 rollout 的轨迹
- 经验回放缓冲仅在 rollout 组中无成功轨迹时补充（论文策略）

### 异步 Rollout

- 环境池管理：维护固定数量的环境实例，重置和重用而非销毁
- 线程池执行：使用 Python threading 实现异步环境交互
- 失败恢复：环境失败时自动替换（需要实现备用会话机制）

### 混合并行

- 当前实现预留接口，不强制依赖外部框架
- 可通过配置 `hybrid_parallelism` 指定并行策略
- 长序列训练建议调整 batch size 和 gradient accumulation

### 奖励与惩罚

- 任务完成奖励：基于上游 `get_task_score` 的二进制奖励
- 动作重复惩罚：检测单步重复和 3-5 步循环模式
- 验证器支持：预留规则验证和 MLLM-as-a-Judge 接口

## 验证与回归

### 训练侧

- ✅ 配置驱动可启动
- ✅ 数据构建可输出标准 JSONL
- ✅ RL rollout 可生成符合上游轨迹格式的 `traj.json`
- ✅ 动作格式与上游完全对齐

### 评测侧

- ✅ 使用上游 runner，确保动作解析一致
- ✅ 支持 MCP 工具调用
- ✅ 支持 `ask_user` 交互
- ⚠️ Zoom-In 策略：当前评估脚本使用上游 runner，Zoom-In 需在 agent 层面实现

## 依赖说明

### 必需依赖

- `torch >= 2.0.0`
- `transformers >= 4.35.0`
- `datasets >= 2.14.0`
- `trl >= 0.7.0`
- `pyyaml >= 6.0`
- `pillow >= 10.0.0`
- `tqdm >= 4.66.0`

### 上游依赖

训练脚本直接使用 `trainer/mobile_world` 目录下的代码，无需额外配置。

---

## 详细文档索引

| 文档 | 说明 |
|-----|------|
| [DATA_FORMAT.md](docs/DATA_FORMAT.md) | 数据格式详细说明（包含三种格式的完整定义） |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | 常见问题排查指南（训练失败、数据错误等） |
| [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md) | 工程优化方案完整文档 |

---

## 故障排查

遇到问题？请先查阅 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)，包含以下常见问题的解决方案：

- 🔧 数据处理失败（图片路径错误、格式不合法）
- 🔧 训练 OOM（显存不足）
- 🔧 RL 训练环境失败
- 🔧 评估结果不一致
- 🔧 Checkpoint 加载失败

如果文档未覆盖您的问题，请提交 Issue 并附上错误日志。

---

## 参考资料

- MAI-UI 技术报告: `../MAI-UI 技术报告： 面向现实世界的基础型 GUI 智能体_2025.12_通义实验室.pdf`
- 上游代码: `../upstream/src/mobile_world/`
- verl 文档: https://verl.readthedocs.io/

---

## 版本历史

### v2.0 (2026-01-19) - 企业级工程化版本
- ✨ 新增统一数据处理器，支持多种输出格式
- ✨ 新增端到端 Pipeline 编排器，支持断点续传
- ✨ 新增批量评估系统，自动生成性能对比报告
- ✨ 新增工具脚本（quick_start.sh、validate_data.py 等）
- ✨ 新增详细文档（DATA_FORMAT.md、TROUBLESHOOTING.md）
- 🔧 优化数据验证和错误恢复机制
- 🔧 统一配置文件，简化使用流程

### v1.0 (2025-12-20) - 基础版本
- 实现 SFT + RL 训练流程
- 对齐上游 MobileWorld 代码库
- 实现基于 verl 的异步 RL 训练

---

## 致谢

感谢 MAI-UI 团队（阿里巴巴通义实验室）提供的详细技术报告和开源模型。

---

## License

本项目遵循 Apache 2.0 许可证。详见 [LICENSE](../LICENSE) 文件。
