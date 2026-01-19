#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
verl-based Trainer for MAI-UI.

Main entry point for training MAI-UI models using verl framework
with MobileWorld environment integration.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from mobile_world.runtime.client import discover_backends
from mobile_world.runtime.utils.trajectory_logger import TrajLogger

from verl_integration.custom_grpo import CustomGRPOTrainer
from verl_integration.hybrid_verifier import HybridVerifier
from verl_integration.iterative_rejection_sampling import IterativeRejectionSampling
from verl_integration.mobile_world_agent_loop import MobileWorldAgentLoop
from verl_integration.reward_calculator import MAIUIRewardCalculator
from verl_integration.task_curriculum import TaskCurriculum


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_mobile_world_environments(config: dict[str, Any]) -> list[str]:
    """Setup MobileWorld environments.
    
    Args:
        config: Configuration dict
    
    Returns:
        List of environment URLs
    """
    env_config = config.get("mobile_world", {}).get("environment", {})
    aw_urls = env_config.get("aw_urls", [])
    
    if not aw_urls:
        # Auto-discover environments
        logger.info("Auto-discovering MobileWorld environments...")
        aw_urls, _ = discover_backends(
            image_filter=env_config.get("env_image", "mobile_world"),
            prefix=env_config.get("env_name_prefix", "mobile_world_env"),
        )
        logger.info(f"Discovered {len(aw_urls)} environments")
    
    num_envs = env_config.get("num_parallel_envs", 32)
    return aw_urls[:num_envs]


def setup_task_curriculum(config: dict[str, Any]) -> TaskCurriculum | None:
    """Setup task curriculum learning.
    
    Args:
        config: Configuration dict
    
    Returns:
        TaskCurriculum instance or None if disabled
    """
    curriculum_config = config.get("curriculum", {})
    if not curriculum_config.get("enabled", True):
        return None
    
    curriculum = TaskCurriculum(
        difficulty_levels=curriculum_config.get("difficulty_levels"),
        min_attempts_per_task=curriculum_config.get("min_attempts_per_task", 5),
        initial_distribution=curriculum_config.get("initial_distribution"),
    )
    
    return curriculum


def setup_reward_calculator(config: dict[str, Any]) -> MAIUIRewardCalculator:
    """Setup reward calculator.
    
    Args:
        config: Configuration dict
    
    Returns:
        MAIUIRewardCalculator instance
    """
    reward_config = config.get("reward", {})
    return MAIUIRewardCalculator(
        repetition_penalty=reward_config.get("repetition_penalty", 0.1),
        cycle_penalty_multiplier=reward_config.get("cycle_penalty_multiplier", 2.0),
        min_reward=reward_config.get("min_reward", 0.0),
    )


def setup_hybrid_verifier(config: dict[str, Any]) -> HybridVerifier:
    """Setup hybrid verifier.
    
    Args:
        config: Configuration dict
    
    Returns:
        HybridVerifier instance
    """
    mw_config = config.get("mobile_world", {})
    return HybridVerifier(
        judge_model=config.get("fine_grained_analysis", {}).get("judge_model", "gpt-4"),
        judge_api_key=config.get("fine_grained_analysis", {}).get("judge_api_key"),
        prefer_rule_verification=True,
    )


def setup_grpo_trainer(config: dict[str, Any]) -> CustomGRPOTrainer:
    """Setup custom GRPO trainer.
    
    Args:
        config: Configuration dict
    
    Returns:
        CustomGRPOTrainer instance
    """
    algorithm_config = config.get("algorithm", {})
    return CustomGRPOTrainer(
        eps_low=algorithm_config.get("clip_eps_low", 0.2),
        eps_high=algorithm_config.get("clip_eps_high", 0.3),
        normalize_by_std=algorithm_config.get("norm_adv_by_std_in_grpo", True),
        kl_coef=config.get("actor_rollout_ref", {}).get("actor", {}).get("kl_loss_coef", 0.001),
    )


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train MAI-UI models using verl framework"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to verl_config.yaml",
    )
    parser.add_argument(
        "--llm_base_url",
        type=str,
        help="LLM base URL (overrides config)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="API key (overrides config)",
    )
    parser.add_argument(
        "--sft_model_path",
        type=str,
        help="SFT model path (overrides config)",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with CLI args
    if args.llm_base_url:
        config["mobile_world"]["llm_base_url"] = args.llm_base_url
    if args.api_key:
        config["mobile_world"]["api_key"] = args.api_key
    if args.sft_model_path:
        config["actor_rollout_ref"]["model"]["path"] = args.sft_model_path
    
    # Validate required fields
    mw_config = config.get("mobile_world", {})
    if not mw_config.get("llm_base_url"):
        raise ValueError(
            "llm_base_url must be provided in config or via --llm_base_url"
        )
    
    # Setup components
    logger.info("Setting up verl training components...")
    
    # Setup environments
    env_urls = setup_mobile_world_environments(config)
    logger.info(f"Using {len(env_urls)} MobileWorld environments")
    
    # Setup task curriculum
    curriculum = setup_task_curriculum(config)
    if curriculum:
        logger.info("Task curriculum learning enabled")
    
    # Setup reward calculator
    reward_calculator = setup_reward_calculator(config)
    
    # Setup hybrid verifier
    verifier = setup_hybrid_verifier(config)
    
    # Setup GRPO trainer
    grpo_trainer = setup_grpo_trainer(config)
    
    # Update verl config with GRPO settings
    config = grpo_trainer.update_config_for_verl(config)
    
    logger.info("Components setup complete")
    logger.info("Note: Full verl integration requires verl framework installation")
    logger.info("Install with: pip install verl>=0.5.0")
    logger.info("")
    logger.info("Configuration summary:")
    logger.info(f"  - GRPO group_size: {config.get('actor_rollout_ref', {}).get('rollout', {}).get('n', 16)}")
    logger.info(f"  - Asymmetric clipping: eps_low={config.get('algorithm', {}).get('clip_eps_low', 0.2)}, eps_high={config.get('algorithm', {}).get('clip_eps_high', 0.3)}")
    logger.info(f"  - Parallel environments: {len(env_urls)}")
    logger.info(f"  - Curriculum learning: {'enabled' if curriculum else 'disabled'}")
    
    # TODO: Integrate with verl's actual training loop
    # This would require:
    # 1. Creating verl RLTrainer instance
    # 2. Setting up AgentLoopManager with MobileWorldAgentLoop
    # 3. Running verl training loop
    # 
    # Example (pseudo-code):
    # from verl import RLTrainer, AgentLoopManager
    # 
    # trainer = RLTrainer(config)
    # agent_loop_manager = AgentLoopManager(
    #     agent_loop_factory=lambda: MobileWorldAgentLoop(...),
    #     ...
    # )
    # trainer.train(agent_loop_manager)
    
    logger.warning(
        "verl integration scaffolding complete. "
        "Full integration requires verl framework and proper AgentLoopManager setup."
    )


if __name__ == "__main__":
    main()
