#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MobileWorld AgentLoop for verl Integration.

Wraps MobileWorld environment as verl's AgentLoop interface,
enabling agentic RL training with verl framework.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from mobile_world.agents.registry import create_agent
from mobile_world.core.runner import _execute_single_task
from mobile_world.runtime.client import (
    AndroidEnvClient,
    AndroidMCPEnvClient,
    discover_backends,
)
from mobile_world.runtime.utils.models import ENV_FAIL, FINISHED, UNKNOWN, ANSWER
from mobile_world.runtime.utils.trajectory_logger import TrajLogger

from .reward_calculator import MAIUIRewardCalculator
from .task_curriculum import TaskCurriculum


class MobileWorldAgentLoop:
    """AgentLoop implementation for MobileWorld environment.
    
    Implements verl's AgentLoop interface for multi-turn GUI agent interactions.
    Each GUI step is treated as one interaction turn.
    """
    
    def __init__(
        self,
        env: AndroidEnvClient,
        agent_type: str,
        model_name: str,
        llm_base_url: str,
        api_key: str,
        max_step: int = 50,
        enable_mcp: bool = False,
        log_file_root: str = "./rl_logs",
        task_curriculum: TaskCurriculum | None = None,
        reward_calculator: MAIUIRewardCalculator | None = None,
        **agent_kwargs,
    ):
        """Initialize MobileWorld AgentLoop.
        
        Args:
            env: Android environment client
            agent_type: Agent type name
            model_name: Model name for agent
            llm_base_url: LLM API base URL
            api_key: API key
            max_step: Maximum steps per trajectory
            enable_mcp: Enable MCP tools
            log_file_root: Root directory for trajectory logs
            task_curriculum: Optional task curriculum for sampling
            reward_calculator: Optional reward calculator
            **agent_kwargs: Additional agent kwargs
        """
        self.env = env
        self.agent_type = agent_type
        self.model_name = model_name
        self.llm_base_url = llm_base_url
        self.api_key = api_key
        self.max_step = max_step
        self.enable_mcp = enable_mcp
        self.log_file_root = log_file_root
        self.agent_kwargs = agent_kwargs
        
        self.task_curriculum = task_curriculum
        self.reward_calculator = reward_calculator or MAIUIRewardCalculator()
        
        # Current interaction state
        self.current_task_name: str | None = None
        self.current_task_goal: str | None = None
        self.current_agent: Any = None
        self.current_traj_logger: TrajLogger | None = None
        self.trajectory: list[dict[str, Any]] = []
        self.current_step = 0
        self.interaction_id: str | None = None
    
    def start_interaction(
        self,
        interaction_id: str,
        task_name: str | None = None,
        difficulty_level: str | None = None,
    ) -> dict[str, Any]:
        """Start a new interaction (task).
        
        Args:
            interaction_id: Unique interaction ID
            task_name: Optional specific task name (if None, sample from curriculum)
            difficulty_level: Optional difficulty level for curriculum sampling
        
        Returns:
            Initial observation dict
        """
        self.interaction_id = interaction_id
        
        # Sample task if not provided
        if task_name is None:
            if self.task_curriculum:
                task_name = self.task_curriculum.sample_task(difficulty_level)
            else:
                # Fallback: sample from available tasks
                task_list = self.env.get_suite_task_list(enable_mcp=self.enable_mcp)
                if not task_list:
                    raise ValueError("No tasks available")
                import random
                task_name = random.choice(task_list)
        
        self.current_task_name = task_name
        
        # Initialize task
        self.current_task_goal = self.env.get_task_goal(task_type=task_name)
        obs = self.env.initialize_task(task_name=task_name)
        
        # Create agent
        self.current_agent = create_agent(
            self.agent_type,
            self.model_name,
            self.llm_base_url,
            self.api_key,
            env=self.env,
            **self.agent_kwargs,
        )
        self.current_agent.initialize(self.current_task_goal)
        
        # Create trajectory logger
        self.current_traj_logger = TrajLogger(self.log_file_root, task_name)
        if self.enable_mcp:
            self.current_traj_logger.log_tools(self.env.tools)
        
        # Reset trajectory
        self.trajectory = []
        self.current_step = 0
        
        # Return initial observation
        return {
            "screenshot": obs.screenshot,
            "tool_call": obs.tool_call,
            "ask_user_response": obs.ask_user_response,
            "task_goal": self.current_task_goal,
            "task_name": task_name,
        }
    
    def generate_response(
        self,
        interaction_id: str,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate agent response for current observation.
        
        Args:
            interaction_id: Interaction ID
            observation: Current observation (screenshot, tool_call, etc.)
        
        Returns:
            Response dict with prediction and action
        """
        if interaction_id != self.interaction_id:
            raise ValueError(f"Interaction ID mismatch: {interaction_id}")
        
        if self.current_step >= self.max_step:
            return {
                "prediction": "Max steps reached",
                "action": {"action_type": "finished", "goal_status": "infeasible"},
                "terminate": True,
            }
        
        self.current_step += 1
        
        # Agent prediction
        prediction, action = self.current_agent.predict(observation)
        
        # Log trajectory step
        if self.current_traj_logger:
            from mobile_world.runtime.utils.models import Observation
            
            obs_obj = Observation(
                screenshot=observation.get("screenshot"),
                tool_call=observation.get("tool_call"),
                ask_user_response=observation.get("ask_user_response"),
            )
            
            self.current_traj_logger.log_traj(
                self.current_task_name,
                self.current_task_goal,
                self.current_step,
                prediction or "",
                action.model_dump(exclude_none=True),
                obs_obj,
                self.current_agent.get_total_token_usage(),
            )
        
        # Store in trajectory
        self.trajectory.append({
            "step": self.current_step,
            "prediction": prediction,
            "action": action.model_dump(exclude_none=True),
            "observation": observation,
        })
        
        # Check termination
        terminate = False
        if prediction is None:
            logger.warning(f"Agent prediction failed in step {self.current_step}")
            terminate = True
        elif action.action_type in [ENV_FAIL, FINISHED, UNKNOWN]:
            terminate = True
        elif action.action_type == ANSWER:
            terminate = True
        
        return {
            "prediction": prediction,
            "action": action.model_dump(exclude_none=True),
            "terminate": terminate,
        }
    
    def execute_action(
        self,
        interaction_id: str,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute action and get next observation.
        
        Args:
            interaction_id: Interaction ID
            action: Action dict
        
        Returns:
            Next observation dict
        """
        if interaction_id != self.interaction_id:
            raise ValueError(f"Interaction ID mismatch: {interaction_id}")
        
        from mobile_world.runtime.utils.models import JSONAction
        
        action_obj = JSONAction(**action)
        obs = self.env.execute_action(action_obj)
        
        return {
            "screenshot": obs.screenshot,
            "tool_call": obs.tool_call,
            "ask_user_response": obs.ask_user_response,
        }
    
    def calculate_score(
        self,
        interaction_id: str,
        trajectory: list[dict[str, Any]] | None = None,
    ) -> tuple[float, str]:
        """Calculate reward/score for the interaction.
        
        Args:
            interaction_id: Interaction ID
            trajectory: Optional trajectory (uses current if None)
        
        Returns:
            Tuple of (score, reason)
        """
        if interaction_id != self.interaction_id:
            raise ValueError(f"Interaction ID mismatch: {interaction_id}")
        
        if trajectory is None:
            trajectory = self.trajectory
        
        # Get task score from environment
        score, reason = self.env.get_task_score(task_type=self.current_task_name)
        
        # Apply reward calculator (includes repetition penalty)
        if self.reward_calculator:
            reward = self.reward_calculator.compute_reward(score, trajectory)
        else:
            reward = score
        
        return reward, reason
    
    def finalize_interaction(
        self,
        interaction_id: str,
        score: float,
        reason: str,
    ) -> dict[str, Any]:
        """Finalize interaction and return results.
        
        Args:
            interaction_id: Interaction ID
            score: Final score
            reason: Score reason
        
        Returns:
            Final interaction result dict
        """
        if interaction_id != self.interaction_id:
            raise ValueError(f"Interaction ID mismatch: {interaction_id}")
        
        # Log final score
        if self.current_traj_logger:
            self.current_traj_logger.log_score(score=score, reason=reason)
        
        # Tear down task
        self.env.tear_down_task(task_type=self.current_task_name)
        if self.current_agent:
            self.current_agent.done()
        
        # Update curriculum if available
        if self.task_curriculum and self.current_task_name:
            success = score > 0.99
            self.task_curriculum.update_success_rate(
                self.current_task_name, success, step=self.current_step
            )
        
        result = {
            "interaction_id": interaction_id,
            "task_name": self.current_task_name,
            "task_goal": self.current_task_goal,
            "score": score,
            "reason": reason,
            "steps": self.current_step,
            "trajectory": self.trajectory,
            "success": score > 0.99,
        }
        
        # Reset state
        self.current_task_name = None
        self.current_task_goal = None
        self.current_agent = None
        self.current_traj_logger = None
        self.trajectory = []
        self.current_step = 0
        self.interaction_id = None
        
        return result
    
    def run_full_interaction(
        self,
        interaction_id: str,
        task_name: str | None = None,
        difficulty_level: str | None = None,
    ) -> dict[str, Any]:
        """Run a complete interaction from start to finish.
        
        Convenience method that combines start_interaction, generate_response,
        execute_action, calculate_score, and finalize_interaction.
        
        Args:
            interaction_id: Interaction ID
            task_name: Optional task name
            difficulty_level: Optional difficulty level
        
        Returns:
            Final interaction result
        """
        # Start interaction
        obs = self.start_interaction(interaction_id, task_name, difficulty_level)
        
        # Main loop
        while True:
            # Generate response
            response = self.generate_response(interaction_id, obs)
            
            if response.get("terminate"):
                break
            
            # Execute action
            obs = self.execute_action(interaction_id, response["action"])
            
            if self.current_step >= self.max_step:
                break
        
        # Calculate score
        score, reason = self.calculate_score(interaction_id)
        
        # Finalize
        result = self.finalize_interaction(interaction_id, score, reason)
        
        return result
