#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Loop Detector for GUI Agent Inference.

This module provides runtime loop detection to prevent the agent from
getting stuck in repetitive action sequences.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionRecord:
    """Record of a single action."""
    step: int
    action_type: str
    action_details: dict
    screenshot_hash: str | None = None  # Optional: hash of screenshot for change detection


@dataclass
class LoopDetectorConfig:
    """Configuration for loop detection."""
    # How many consecutive same actions trigger detection
    same_action_threshold: int = 3
    
    # How many steps to check for cyclic patterns
    cycle_detection_window: int = 10
    
    # Minimum cycle length to detect
    min_cycle_length: int = 2
    
    # Whether to use screenshot hashing for change detection
    use_screenshot_hash: bool = False
    
    # How many same screenshots trigger "no change" detection
    same_screenshot_threshold: int = 2


class LoopDetector:
    """Detects and prevents action loops during GUI Agent inference."""
    
    def __init__(self, config: LoopDetectorConfig | None = None):
        self.config = config or LoopDetectorConfig()
        self.history: deque[ActionRecord] = deque(
            maxlen=self.config.cycle_detection_window
        )
        self.step_count = 0
        self.intervention_count = 0
    
    def record_action(
        self,
        action_type: str,
        action_details: dict,
        screenshot_hash: str | None = None,
    ) -> None:
        """Record an action to history."""
        self.step_count += 1
        record = ActionRecord(
            step=self.step_count,
            action_type=action_type,
            action_details=action_details,
            screenshot_hash=screenshot_hash,
        )
        self.history.append(record)
    
    def check_same_action_loop(self) -> tuple[bool, int]:
        """Check if the same action has been repeated consecutively.
        
        Returns:
            (is_looping, repeat_count)
        """
        if len(self.history) < self.config.same_action_threshold:
            return False, 0
        
        recent = list(self.history)[-self.config.same_action_threshold:]
        action_types = [r.action_type for r in recent]
        
        if len(set(action_types)) == 1:
            # Count total consecutive same actions
            count = 0
            for r in reversed(list(self.history)):
                if r.action_type == action_types[0]:
                    count += 1
                else:
                    break
            return True, count
        
        return False, 0
    
    def check_cyclic_pattern(self) -> tuple[bool, list[str], int]:
        """Check for cyclic patterns (e.g., A->B->A->B->A->B).
        
        Returns:
            (has_cycle, cycle_pattern, cycle_count)
        """
        if len(self.history) < self.config.min_cycle_length * 2:
            return False, [], 0
        
        actions = [r.action_type for r in self.history]
        
        # Check for cycles of different lengths
        for cycle_len in range(
            self.config.min_cycle_length,
            len(actions) // 2 + 1
        ):
            # Get potential cycle
            potential_cycle = actions[-cycle_len:]
            
            # Check how many times this cycle repeats
            repeats = 0
            pos = len(actions) - cycle_len
            
            while pos >= cycle_len:
                check_range = actions[pos - cycle_len:pos]
                if check_range == potential_cycle:
                    repeats += 1
                    pos -= cycle_len
                else:
                    break
            
            if repeats >= 2:  # Cycle repeated at least twice
                return True, potential_cycle, repeats + 1
        
        return False, [], 0
    
    def check_no_screen_change(self) -> tuple[bool, int]:
        """Check if the screen hasn't changed despite actions.
        
        Returns:
            (no_change, same_count)
        """
        if not self.config.use_screenshot_hash:
            return False, 0
        
        if len(self.history) < self.config.same_screenshot_threshold:
            return False, 0
        
        recent = list(self.history)[-self.config.same_screenshot_threshold:]
        hashes = [r.screenshot_hash for r in recent if r.screenshot_hash]
        
        if len(hashes) < self.config.same_screenshot_threshold:
            return False, 0
        
        if len(set(hashes)) == 1:
            # Count total consecutive same screenshots
            count = 0
            last_hash = hashes[-1]
            for r in reversed(list(self.history)):
                if r.screenshot_hash == last_hash:
                    count += 1
                else:
                    break
            return True, count
        
        return False, 0
    
    def should_intervene(self) -> tuple[bool, str, dict]:
        """Check if intervention is needed.
        
        Returns:
            (should_intervene, reason, details)
        """
        # Check same action loop
        same_loop, same_count = self.check_same_action_loop()
        if same_loop and same_count >= self.config.same_action_threshold:
            return True, "same_action_loop", {
                "action_type": list(self.history)[-1].action_type,
                "repeat_count": same_count,
            }
        
        # Check cyclic pattern
        has_cycle, pattern, cycle_count = self.check_cyclic_pattern()
        if has_cycle:
            return True, "cyclic_pattern", {
                "pattern": pattern,
                "cycle_count": cycle_count,
            }
        
        # Check no screen change
        no_change, change_count = self.check_no_screen_change()
        if no_change and change_count >= self.config.same_screenshot_threshold:
            return True, "no_screen_change", {
                "same_count": change_count,
            }
        
        return False, "", {}
    
    def get_intervention_action(self, reason: str, details: dict) -> dict:
        """Get the intervention action to break the loop.
        
        Returns:
            Action dict that should replace the model's output
        """
        self.intervention_count += 1
        
        if reason == "same_action_loop":
            action_type = details.get("action_type", "unknown")
            repeat_count = details.get("repeat_count", 0)
            
            if action_type == "swipe":
                return {
                    "action": "ask_user",
                    "text": f"我已经连续滑动了 {repeat_count} 次，但没有找到目标。"
                            "请问：1) 目标大概在什么位置？2) 有没有可以用来搜索的关键词？",
                    "_intervention": True,
                    "_reason": f"Detected {repeat_count} consecutive swipe actions",
                }
            elif action_type == "click":
                return {
                    "action": "wait",
                    "_intervention": True,
                    "_reason": f"Detected {repeat_count} consecutive click actions, waiting for page load",
                }
            else:
                return {
                    "action": "ask_user",
                    "text": f"当前操作（{action_type}）似乎没有效果，请问是否需要调整策略？",
                    "_intervention": True,
                    "_reason": f"Detected {repeat_count} consecutive {action_type} actions",
                }
        
        elif reason == "cyclic_pattern":
            pattern = details.get("pattern", [])
            cycle_count = details.get("cycle_count", 0)
            return {
                "action": "ask_user",
                "text": f"我发现操作陷入了循环（{' -> '.join(pattern)}，重复 {cycle_count} 次）。"
                        "请问任务是否需要调整？",
                "_intervention": True,
                "_reason": f"Detected cyclic pattern: {pattern}",
            }
        
        elif reason == "no_screen_change":
            same_count = details.get("same_count", 0)
            return {
                "action": "ask_user",
                "text": f"屏幕内容已经 {same_count} 步没有变化，可能已经到达边界或操作无效。请确认下一步操作。",
                "_intervention": True,
                "_reason": f"Screen unchanged for {same_count} steps",
            }
        
        # Fallback
        return {
            "action": "ask_user",
            "text": "当前操作模式异常，请确认下一步操作。",
            "_intervention": True,
            "_reason": reason,
        }
    
    def reset(self) -> None:
        """Reset the detector state."""
        self.history.clear()
        self.step_count = 0
    
    def get_stats(self) -> dict:
        """Get statistics about the current session."""
        return {
            "total_steps": self.step_count,
            "intervention_count": self.intervention_count,
            "history_length": len(self.history),
        }


def integrate_with_agent(
    agent_output: dict,
    detector: LoopDetector,
    screenshot_hash: str | None = None,
) -> dict:
    """Integrate loop detector with agent output.
    
    Args:
        agent_output: The action output from the GUI agent
        detector: The loop detector instance
        screenshot_hash: Optional hash of current screenshot
    
    Returns:
        Either the original agent output, or an intervention action
    """
    action_type = agent_output.get("action", "unknown")
    
    # Check if intervention is needed BEFORE recording
    should_intervene, reason, details = detector.should_intervene()
    
    if should_intervene:
        intervention = detector.get_intervention_action(reason, details)
        # Record the intervention action instead
        detector.record_action(
            intervention["action"],
            intervention,
            screenshot_hash,
        )
        return intervention
    
    # Record the action
    detector.record_action(action_type, agent_output, screenshot_hash)
    
    return agent_output


# Example usage
if __name__ == "__main__":
    # Create detector with default config
    config = LoopDetectorConfig(
        same_action_threshold=3,
        cycle_detection_window=10,
    )
    detector = LoopDetector(config)
    
    # Simulate a loop scenario
    actions = [
        {"action": "swipe", "direction": "up"},
        {"action": "swipe", "direction": "up"},
        {"action": "swipe", "direction": "up"},
        {"action": "swipe", "direction": "up"},  # Should trigger intervention
    ]
    
    print("Simulating action sequence:")
    for i, action in enumerate(actions):
        result = integrate_with_agent(action, detector)
        print(f"Step {i+1}: Input={action['action']}, Output={result['action']}")
        if result.get("_intervention"):
            print(f"  -> INTERVENTION: {result.get('_reason')}")
            print(f"  -> Message: {result.get('text', 'N/A')}")
    
    print(f"\nStats: {detector.get_stats()}")
