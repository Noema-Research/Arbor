#!/usr/bin/env python3
"""
Task Planning and Reasoning for Arbor Agents.

This module provides advanced planning capabilities including:
- Multi-step task decomposition
- Reasoning chain management
- Goal-oriented planning
- Adaptive execution strategies
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Represents a single task in a plan."""
    id: str
    description: str
    tool_name: str
    arguments: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""
    reasoning: str = ""


@dataclass
class ExecutionPlan:
    """Represents a complete execution plan."""
    goal: str
    tasks: List[Task]
    reasoning: str = ""
    estimated_time: float = 0.0
    complexity_score: float = 0.0


class ReasoningChain:
    """Manages reasoning chain for complex tasks."""
    
    def __init__(self):
        self.steps: List[str] = []
        self.confidence_scores: List[float] = []
        self.logger = logging.getLogger(f"{__class__.__name__}")
    
    def add_step(self, step: str, confidence: float = 1.0):
        """Add a reasoning step."""
        self.steps.append(step)
        self.confidence_scores.append(confidence)
        self.logger.debug(f"💭 Reasoning step: {step} (confidence: {confidence:.2f})")
    
    def get_chain(self) -> str:
        """Get the complete reasoning chain."""
        chain_parts = []
        for i, (step, confidence) in enumerate(zip(self.steps, self.confidence_scores)):
            chain_parts.append(f"{i+1}. {step} (confidence: {confidence:.2f})")
        
        return "\n".join(chain_parts)
    
    def get_average_confidence(self) -> float:
        """Get average confidence score."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)


class TaskPlanner:
    """Advanced task planner for multi-step reasoning."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__class__.__name__}")
        self.execution_history: List[ExecutionPlan] = []
    
    async def create_plan(
        self,
        goal: str,
        available_tools: List[str],
        context: Dict[str, Any] = None
    ) -> ExecutionPlan:
        """
        Create an execution plan for achieving a goal.
        
        Args:
            goal: The goal to achieve
            available_tools: List of available tool names
            context: Additional context information
            
        Returns:
            ExecutionPlan with tasks to execute
        """
        self.logger.info(f"🎯 Creating plan for goal: {goal}")
        
        # Analyze goal complexity
        complexity = self._analyze_complexity(goal)
        
        # Decompose goal into tasks
        tasks = await self._decompose_goal(goal, available_tools, context)
        
        # Estimate execution time
        estimated_time = self._estimate_execution_time(tasks)
        
        # Create reasoning for the plan
        reasoning = await self._generate_plan_reasoning(goal, tasks)
        
        plan = ExecutionPlan(
            goal=goal,
            tasks=tasks,
            reasoning=reasoning,
            estimated_time=estimated_time,
            complexity_score=complexity
        )
        
        self.logger.info(f"📋 Created plan with {len(tasks)} tasks (complexity: {complexity:.2f})")
        return plan
    
    def _analyze_complexity(self, goal: str) -> float:
        """Analyze goal complexity (0.0 to 1.0)."""
        complexity_indicators = [
            ("multiple", 0.3),
            ("complex", 0.4),
            ("analyze", 0.3),
            ("create", 0.2),
            ("generate", 0.2),
            ("compare", 0.3),
            ("optimize", 0.4),
            ("implement", 0.5),
            ("debug", 0.4),
            ("research", 0.4),
            ("evaluate", 0.3)
        ]
        
        goal_lower = goal.lower()
        complexity = 0.1  # Base complexity
        
        for indicator, weight in complexity_indicators:
            if indicator in goal_lower:
                complexity += weight
        
        # Word count factor
        word_count = len(goal.split())
        if word_count > 10:
            complexity += 0.2
        if word_count > 20:
            complexity += 0.2
        
        return min(1.0, complexity)
    
    async def _decompose_goal(
        self,
        goal: str,
        available_tools: List[str],
        context: Dict[str, Any] = None
    ) -> List[Task]:
        """Decompose goal into executable tasks."""
        tasks = []
        
        # Simple rule-based decomposition
        # In a real implementation, this could use LLM-based planning
        
        goal_lower = goal.lower()
        
        # Code-related goals
        if any(keyword in goal_lower for keyword in ["code", "program", "script", "function"]):
            if "python_code" in available_tools:
                tasks.append(Task(
                    id="code_1",
                    description="Write Python code to solve the problem",
                    tool_name="python_code",
                    arguments={"code": f"# Code for: {goal}\nprint('Implementation needed')"}
                ))
        
        # File-related goals
        if any(keyword in goal_lower for keyword in ["file", "read", "write", "save"]):
            if "read" in goal_lower and "read_file" in available_tools:
                tasks.append(Task(
                    id="file_read_1",
                    description="Read required file",
                    tool_name="read_file",
                    arguments={"file_path": "input.txt"}  # Would be extracted from goal
                ))
            
            if "write" in goal_lower and "write_file" in available_tools:
                tasks.append(Task(
                    id="file_write_1",
                    description="Write output to file",
                    tool_name="write_file",
                    arguments={"file_path": "output.txt", "content": "Generated content"}
                ))
        
        # Search-related goals
        if any(keyword in goal_lower for keyword in ["search", "find", "lookup", "research"]):
            if "web_search" in available_tools:
                # Extract search terms from goal
                search_query = goal.replace("search for", "").replace("find", "").strip()
                tasks.append(Task(
                    id="search_1",
                    description="Search for relevant information",
                    tool_name="web_search",
                    arguments={"query": search_query}
                ))
        
        # Math-related goals
        if any(keyword in goal_lower for keyword in ["calculate", "compute", "math", "equation"]):
            if "calculator" in available_tools:
                tasks.append(Task(
                    id="calc_1",
                    description="Perform mathematical calculation",
                    tool_name="calculator",
                    arguments={"expression": "1 + 1"}  # Would be extracted from goal
                ))
        
        # Default task if no specific patterns matched
        if not tasks:
            # Try to use the most appropriate available tool
            if "python_code" in available_tools:
                tasks.append(Task(
                    id="general_1",
                    description=f"Address the goal: {goal}",
                    tool_name="python_code",
                    arguments={"code": f"# Solving: {goal}\nprint('Task completed')"}
                ))
        
        # Add task dependencies (simple linear dependency)
        for i in range(1, len(tasks)):
            tasks[i].dependencies.append(tasks[i-1].id)
        
        return tasks
    
    def _estimate_execution_time(self, tasks: List[Task]) -> float:
        """Estimate total execution time in seconds."""
        time_estimates = {
            "python_code": 5.0,
            "bash_command": 3.0,
            "read_file": 1.0,
            "write_file": 2.0,
            "web_search": 8.0,
            "calculator": 1.0
        }
        
        total_time = 0.0
        for task in tasks:
            tool_time = time_estimates.get(task.tool_name, 5.0)
            total_time += tool_time
        
        return total_time
    
    async def _generate_plan_reasoning(self, goal: str, tasks: List[Task]) -> str:
        """Generate reasoning for the execution plan."""
        reasoning_parts = [
            f"🎯 Goal: {goal}",
            f"📊 Analysis: Decomposed into {len(tasks)} tasks",
            "",
            "📋 Execution strategy:"
        ]
        
        for i, task in enumerate(tasks, 1):
            dependencies = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
            reasoning_parts.append(f"  {i}. {task.description}{dependencies}")
        
        return "\n".join(reasoning_parts)
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        tool_executor: Callable[[str, Dict[str, Any]], Any]
    ) -> Dict[str, Any]:
        """
        Execute an execution plan.
        
        Args:
            plan: The execution plan to run
            tool_executor: Function to execute tools (tool_name, arguments) -> result
            
        Returns:
            Execution results
        """
        self.logger.info(f"🚀 Executing plan: {plan.goal}")
        
        completed_tasks = set()
        results = {}
        
        while len(completed_tasks) < len(plan.tasks):
            # Find tasks ready to execute
            ready_tasks = [
                task for task in plan.tasks
                if (task.status == TaskStatus.PENDING and
                    all(dep in completed_tasks for dep in task.dependencies))
            ]
            
            if not ready_tasks:
                # Check for failed dependencies
                failed_tasks = [t for t in plan.tasks if t.status == TaskStatus.FAILED]
                if failed_tasks:
                    self.logger.error(f"❌ Plan execution failed due to failed tasks: {[t.id for t in failed_tasks]}")
                    break
                
                # No ready tasks and no failures - possible deadlock
                self.logger.error("❌ Plan execution deadlock - no ready tasks")
                break
            
            # Execute ready tasks
            for task in ready_tasks:
                try:
                    self.logger.info(f"🔧 Executing task: {task.id} - {task.description}")
                    task.status = TaskStatus.IN_PROGRESS
                    
                    # Execute the task
                    result = await tool_executor(task.tool_name, task.arguments)
                    
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    completed_tasks.add(task.id)
                    results[task.id] = result
                    
                    self.logger.info(f"✅ Task completed: {task.id}")
                    
                except Exception as e:
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
                    self.logger.error(f"❌ Task failed: {task.id} - {e}")
        
        # Store execution history
        self.execution_history.append(plan)
        
        execution_summary = {
            "goal": plan.goal,
            "total_tasks": len(plan.tasks),
            "completed_tasks": len(completed_tasks),
            "success_rate": len(completed_tasks) / len(plan.tasks),
            "results": results
        }
        
        self.logger.info(f"📊 Plan execution complete: {len(completed_tasks)}/{len(plan.tasks)} tasks")
        return execution_summary
    
    def get_execution_history(self) -> List[ExecutionPlan]:
        """Get history of executed plans."""
        return self.execution_history.copy()
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze planner performance."""
        if not self.execution_history:
            return {"message": "No execution history available"}
        
        total_plans = len(self.execution_history)
        total_tasks = sum(len(plan.tasks) for plan in self.execution_history)
        
        # Calculate success rates (would need to track actual completions)
        avg_complexity = sum(plan.complexity_score for plan in self.execution_history) / total_plans
        avg_tasks_per_plan = total_tasks / total_plans
        
        return {
            "total_plans_executed": total_plans,
            "total_tasks_planned": total_tasks,
            "average_complexity": avg_complexity,
            "average_tasks_per_plan": avg_tasks_per_plan,
            "most_complex_goal": max(self.execution_history, key=lambda p: p.complexity_score).goal
        }
