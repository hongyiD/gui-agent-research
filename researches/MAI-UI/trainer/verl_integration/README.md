# verl Integration for MAI-UI Training

This module provides verl framework integration for training MAI-UI models with MobileWorld environments.

## Overview

The verl integration implements:

1. **GRPO with Asymmetric Clipping**: Group Relative Policy Optimization with eps_low=0.2, eps_high=0.3
2. **Task Curriculum Learning**: Dynamic task difficulty stratification based on success rates
3. **Fine-grained Trajectory Analysis**: MLLM-as-Judge for extracting correct prefixes from failed trajectories
4. **Hybrid Verification**: Rule-based + MLLM-as-Judge verification (83% human agreement)
5. **Iterative Rejection Sampling**: Self-evolution data pipeline
6. **MobileWorld AgentLoop**: Wrapper for MobileWorld environment as verl AgentLoop

## Installation

Install verl and dependencies:

```bash
pip install verl>=0.5.0
pip install vllm>=0.8.0  # or sglang for async inference
pip install ray>=2.0.0
```

## Configuration

See `configs/verl_config.yaml` for complete configuration options.

Key settings:

- **GRPO group_size**: Set `actor_rollout_ref.rollout.n: 16`
- **Asymmetric clipping**: `algorithm.clip_eps_low: 0.2`, `algorithm.clip_eps_high: 0.3`
- **Async mode**: `actor_rollout_ref.rollout.mode: async`
- **Curriculum learning**: `curriculum.enabled: true`

## Usage

### Basic Training

```bash
python verl_trainer.py \
    --config configs/verl_config.yaml \
    --llm_base_url https://api.openai.com/v1 \
    --api_key your-api-key \
    --sft_model_path /path/to/sft_model
```

### With Custom Configuration

```bash
python verl_trainer.py \
    --config configs/verl_config.yaml \
    --llm_base_url https://api.openai.com/v1 \
    --api_key your-api-key
```

## Components

### MobileWorldAgentLoop

Wraps MobileWorld environment as verl's AgentLoop interface:

```python
from verl_integration.mobile_world_agent_loop import MobileWorldAgentLoop

agent_loop = MobileWorldAgentLoop(
    env=env,
    agent_type="mai_ui_agent",
    model_name="Tongyi-MAI/MAI-UI-2B",
    llm_base_url="https://api.openai.com/v1",
    api_key="your-key",
    max_step=50,
    task_curriculum=curriculum,
    reward_calculator=reward_calc,
)

# Run full interaction
result = agent_loop.run_full_interaction(
    interaction_id="interaction_1",
    task_name="task_name",
)
```

### TaskCurriculum

Manages dynamic task difficulty stratification:

```python
from verl_integration.task_curriculum import TaskCurriculum

curriculum = TaskCurriculum(
    min_attempts_per_task=5,
    initial_distribution={
        "frontier": 0.1,
        "exploration": 0.3,
        "near_mastery": 0.4,
        "exploitation": 0.2,
    },
)

# Register tasks
curriculum.register_tasks(task_list)

# Sample task
task_name = curriculum.sample_task(difficulty_level="exploration")

# Update success rate
curriculum.update_success_rate(task_name, success=True, step=100)
```

### CustomGRPOTrainer

GRPO with asymmetric clipping:

```python
from verl_integration.custom_grpo import CustomGRPOTrainer

grpo_trainer = CustomGRPOTrainer(
    eps_low=0.2,
    eps_high=0.3,
    normalize_by_std=True,
    kl_coef=0.001,
)

# Update verl config
config = grpo_trainer.update_config_for_verl(config)
```

### FineGrainedTrajectoryAnalyzer

Extract correct prefixes from failed trajectories:

```python
from verl_integration.fine_grained_analyzer import FineGrainedTrajectoryAnalyzer

analyzer = FineGrainedTrajectoryAnalyzer(
    judge_model="gpt-4",
    extract_correct_prefix=True,
)

correct_prefix, error_step = analyzer.analyze_and_extract(
    task_goal="Task goal",
    trajectory=trajectory,
)
```

### HybridVerifier

Rule-based + MLLM-as-Judge verification:

```python
from verl_integration.hybrid_verifier import HybridVerifier

verifier = HybridVerifier(
    judge_model="gpt-4",
    prefer_rule_verification=True,
)

score, reason = verifier.verify(
    task_name="task_name",
    task_goal="Task goal",
    trajectory=trajectory,
)
```

## Integration with verl Framework

Full integration requires verl's AgentLoopManager. The current implementation provides:

1. **MobileWorldAgentLoop**: Compatible with verl's AgentLoop interface
2. **Configuration**: verl-compatible YAML configuration
3. **GRPO Extension**: Custom GRPO with asymmetric clipping

To complete integration:

1. Install verl framework
2. Set up verl's AgentLoopManager with MobileWorldAgentLoop factory
3. Configure verl RLTrainer with GRPO algorithm
4. Run verl training loop

## Status

Current implementation provides:
- ✅ All core components (AgentLoop, Curriculum, GRPO, etc.)
- ✅ Configuration files
- ✅ Integration scaffolding
- ⚠️ Full verl framework integration (requires verl installation and AgentLoopManager setup)

## Next Steps

1. Install verl framework: `pip install verl>=0.5.0`
2. Complete AgentLoopManager integration
3. Test with small-scale training run
4. Scale to full training pipeline
