#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom GRPO Implementation for MAI-UI.

Extends verl's GRPO with asymmetric clipping support as described
in the MAI-UI paper (eps_low=0.2, eps_high=0.3).
"""

from __future__ import annotations

import torch
from typing import Any

from loguru import logger


def compute_grpo_loss_with_asymmetric_clip(
    ratios: torch.Tensor,
    advantages: torch.Tensor,
    eps_low: float = 0.2,
    eps_high: float = 0.3,
) -> torch.Tensor:
    """Compute GRPO loss with asymmetric clipping.
    
    Implements the MAI-UI GRPO objective:
    J_GRPO = E[1/sum(|o_c|) * sum(min(r_i,t * A_hat_i,t, 
                                      clip(r_i,t, 1-eps_low, 1+eps_high) * A_hat_i,t))]
    
    Args:
        ratios: Importance sampling ratios (pi_new / pi_old)
        advantages: Normalized advantages (group-relative)
        eps_low: Lower clipping bound (default 0.2)
        eps_high: Upper clipping bound (default 0.3, encourages exploration)
    
    Returns:
        GRPO loss (negative for gradient descent)
    """
    # Compute clipped ratios with asymmetric bounds
    clipped_ratios = torch.clamp(ratios, 1 - eps_low, 1 + eps_high)
    
    # Compute surrogate losses
    surr1 = ratios * advantages
    surr2 = clipped_ratios * advantages
    
    # Take minimum (PPO-style clipping)
    loss = -torch.mean(torch.min(surr1, surr2))
    
    return loss


def compute_group_relative_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    normalize_by_std: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute group-relative advantages for GRPO.
    
    For each group, compute advantages as:
    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)  if normalize_by_std
    A_i = (r_i - mean(r_group))                          otherwise
    
    Args:
        rewards: Reward values for each sample
        group_ids: Group ID for each sample (same ID = same group)
        normalize_by_std: Whether to normalize by standard deviation
        eps: Small epsilon for numerical stability
    
    Returns:
        Normalized advantages
    """
    advantages = torch.zeros_like(rewards)
    unique_groups = torch.unique(group_ids)
    
    for group_id in unique_groups:
        group_mask = group_ids == group_id
        group_rewards = rewards[group_mask]
        
        if len(group_rewards) == 0:
            continue
        
        # Compute group mean
        group_mean = torch.mean(group_rewards)
        
        if normalize_by_std:
            # Normalize by standard deviation
            group_std = torch.std(group_rewards) + eps
            group_advantages = (group_rewards - group_mean) / group_std
        else:
            # Just subtract mean
            group_advantages = group_rewards - group_mean
        
        advantages[group_mask] = group_advantages
    
    return advantages


class CustomGRPOTrainer:
    """Custom GRPO trainer with asymmetric clipping support.
    
    This class can be used to extend verl's GRPO implementation
    with MAI-UI specific features.
    """
    
    def __init__(
        self,
        eps_low: float = 0.2,
        eps_high: float = 0.3,
        normalize_by_std: bool = True,
        kl_coef: float = 0.001,
    ):
        """Initialize custom GRPO trainer.
        
        Args:
            eps_low: Lower clipping bound
            eps_high: Upper clipping bound
            normalize_by_std: Whether to normalize advantages by std
            kl_coef: KL divergence coefficient
        """
        self.eps_low = eps_low
        self.eps_high = eps_high
        self.normalize_by_std = normalize_by_std
        self.kl_coef = kl_coef
    
    def compute_loss(
        self,
        logprobs_new: torch.Tensor,
        logprobs_old: torch.Tensor,
        rewards: torch.Tensor,
        group_ids: torch.Tensor,
        ref_logprobs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute GRPO loss with asymmetric clipping.
        
        Args:
            logprobs_new: Log probabilities under current policy
            logprobs_old: Log probabilities under old policy (for IS ratio)
            rewards: Reward values
            group_ids: Group IDs for GRPO grouping
            ref_logprobs: Reference policy logprobs for KL penalty
        
        Returns:
            Tuple of (loss, stats_dict)
        """
        # Compute importance sampling ratios
        ratios = torch.exp(logprobs_new - logprobs_old)
        
        # Compute group-relative advantages
        advantages = compute_group_relative_advantages(
            rewards, group_ids, normalize_by_std=self.normalize_by_std
        )
        
        # Compute GRPO loss with asymmetric clipping
        policy_loss = compute_grpo_loss_with_asymmetric_clip(
            ratios, advantages, eps_low=self.eps_low, eps_high=self.eps_high
        )
        
        # Add KL penalty if reference policy provided
        kl_loss = torch.tensor(0.0, device=policy_loss.device)
        if ref_logprobs is not None:
            kl_loss = torch.mean(logprobs_new - ref_logprobs)
        
        total_loss = policy_loss + self.kl_coef * kl_loss
        
        # Compute statistics
        stats = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
            "mean_ratio": ratios.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "std_advantage": advantages.std().item(),
        }
        
        return total_loss, stats
    
    def update_config_for_verl(self, verl_config: dict[str, Any]) -> dict[str, Any]:
        """Update verl config to use custom GRPO settings.
        
        Args:
            verl_config: verl configuration dict
        
        Returns:
            Updated configuration
        """
        # Update algorithm config
        if "algorithm" not in verl_config:
            verl_config["algorithm"] = {}
        
        verl_config["algorithm"]["name"] = "grpo"
        verl_config["algorithm"]["norm_adv_by_std_in_grpo"] = self.normalize_by_std
        
        # Note: verl may not support asymmetric clipping natively,
        # so this would need to be implemented via verl extension mechanism
        verl_config["algorithm"]["clip_eps_low"] = self.eps_low
        verl_config["algorithm"]["clip_eps_high"] = self.eps_high
        
        # Update actor config for KL loss
        if "actor_rollout_ref" not in verl_config:
            verl_config["actor_rollout_ref"] = {}
        if "actor" not in verl_config["actor_rollout_ref"]:
            verl_config["actor_rollout_ref"]["actor"] = {}
        
        verl_config["actor_rollout_ref"]["actor"]["use_kl_loss"] = True
        verl_config["actor_rollout_ref"]["actor"]["kl_loss_coef"] = self.kl_coef
        
        return verl_config
