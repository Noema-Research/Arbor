#!/usr/bin/env python3
"""
Arbor Base Agent Implementation.

This module provides the core agentic capabilities for Arbor including:
- Tool calling and execution
- Multi-step reasoning
- Code execution
- MCP integration
"""

import asyncio
import json
import logging
import traceback
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..modeling.model import ArborTransformer
from ..modeling.multimodal import MultimodalArborTransformer


@dataclass
class AgentResponse:
    """Response from an agent action."""
    content: str
    tool_calls: List[Dict[str, Any]] = None
    reasoning: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = None


@dataclass 
class ToolCall:
    """Represents a tool call request."""
    name: str
    arguments: Dict[str, Any]
    call_id: str = None


class ArborAgent:
    """
    Main Arbor Agent with agentic capabilities.
    
    Features:
    - Code execution (Python, Bash, SQL)
    - Tool calling and function execution
    - Multi-step reasoning and planning
    - MCP (Model Context Protocol) integration
    - Memory and context management
    """
    
    def __init__(
        self,
        model_name_or_path: str = "Noema-Research/arbor-base",
        multimodal: bool = False,
        tools: List["BaseTool"] = None,
        max_context: int = 131072,
        device: str = "auto"
    ):
        """Initialize the Arbor Agent."""
        self.logger = logging.getLogger(f"{__class__.__name__}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if multimodal:
            self.model = MultimodalArborTransformer.from_pretrained(model_name_or_path)
        else:
            self.model = ArborTransformer.from_pretrained(model_name_or_path)
            
        # Setup device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        
        # Agent configuration
        self.max_context = max_context
        self.conversation_history = []
        self.tool_registry = ToolRegistry()
        self.code_executor = CodeExecutor()
        self.task_planner = TaskPlanner()
        
        # Register default tools
        self._register_default_tools()
        
        # Register custom tools
        if tools:
            for tool in tools:
                self.tool_registry.register(tool)
                
        self.logger.info(f"🌳 Arbor Agent initialized with {len(self.tool_registry.tools)} tools")
    
    def _register_default_tools(self):
        """Register default tools for the agent."""
        from .tools import (
            PythonCodeTool, BashCommandTool, FileReadTool, 
            FileWriteTool, WebSearchTool, CalculatorTool
        )
        
        default_tools = [
            PythonCodeTool(),
            BashCommandTool(),
            FileReadTool(),
            FileWriteTool(),
            WebSearchTool(),
            CalculatorTool()
        ]
        
        for tool in default_tools:
            self.tool_registry.register(tool)
    
    async def chat(
        self,
        message: str,
        system_prompt: str = None,
        max_iterations: int = 10,
        temperature: float = 0.7,
        **kwargs
    ) -> AgentResponse:
        """
        Main chat interface with agentic capabilities.
        
        Args:
            message: User message
            system_prompt: Optional system prompt override
            max_iterations: Maximum tool calling iterations
            temperature: Sampling temperature
            
        Returns:
            AgentResponse with final result
        """
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
            
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        iteration = 0
        current_message = message
        reasoning_chain = []
        
        while iteration < max_iterations:
            # Generate response with current context
            response = await self._generate_response(
                current_message,
                system_prompt,
                temperature=temperature,
                **kwargs
            )
            
            # Check if response contains tool calls
            tool_calls = self._extract_tool_calls(response)
            
            if not tool_calls:
                # No more tools needed, return final response
                final_response = AgentResponse(
                    content=response,
                    reasoning="\n".join(reasoning_chain),
                    confidence=self._calculate_confidence(response),
                    metadata={"iterations": iteration + 1}
                )
                
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": response
                })
                
                return final_response
            
            # Execute tool calls
            tool_results = []
            for tool_call in tool_calls:
                try:
                    result = await self._execute_tool_call(tool_call)
                    tool_results.append(result)
                    reasoning_chain.append(f"🔧 Used {tool_call.name}: {result[:100]}...")
                except Exception as e:
                    error_msg = f"❌ Tool {tool_call.name} failed: {str(e)}"
                    tool_results.append(error_msg)
                    reasoning_chain.append(error_msg)
            
            # Prepare next iteration with tool results
            tool_context = "\n".join([
                f"Tool: {tc.name}\nResult: {result}" 
                for tc, result in zip(tool_calls, tool_results)
            ])
            
            current_message = f"""
Previous response: {response}

Tool execution results:
{tool_context}

Continue with the task based on these results.
"""
            
            iteration += 1
        
        # Max iterations reached
        return AgentResponse(
            content="❌ Maximum iterations reached. Task may be incomplete.",
            reasoning="\n".join(reasoning_chain),
            confidence=0.5,
            metadata={"iterations": max_iterations, "incomplete": True}
        )
    
    async def _generate_response(
        self, 
        message: str, 
        system_prompt: str,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate response using Arbor model."""
        # Prepare conversation context
        conversation_text = self._format_conversation(message, system_prompt)
        
        # Tokenize with adaptive context
        inputs = self.tokenizer(
            conversation_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context
        ).to(self.device)
        
        # Generate with model
        with torch.no_grad():
            # Use adaptive context if available
            if hasattr(self.model, 'adaptive_context'):
                with self.model.adaptive_context():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        **kwargs
                    )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _format_conversation(self, message: str, system_prompt: str) -> str:
        """Format conversation for model input."""
        # Build conversation context
        context_parts = [system_prompt]
        
        # Add conversation history
        for turn in self.conversation_history[-10:]:  # Last 10 turns
            role = turn["role"]
            content = turn["content"]
            context_parts.append(f"{role.title()}: {content}")
        
        # Add current message
        context_parts.append(f"User: {message}")
        context_parts.append("Assistant:")
        
        return "\n\n".join(context_parts)
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for agentic behavior."""
        tools_description = self.tool_registry.get_tools_description()
        
        return f"""You are Arbor, an advanced AI agent with access to various tools and capabilities.

You can execute code, search the web, read/write files, and use other tools to help users accomplish tasks.

Available Tools:
{tools_description}

To use a tool, format your response as:
```tool_call
{{
    "name": "tool_name",
    "arguments": {{
        "arg1": "value1",
        "arg2": "value2"
    }}
}}
```

You can make multiple tool calls in a single response. Always explain your reasoning and provide helpful context.

Guidelines:
- Be thorough and accurate
- Explain your thought process
- Use tools when they would be helpful
- Provide clear, actionable responses
- Handle errors gracefully
"""
    
    def _extract_tool_calls(self, response: str) -> List[ToolCall]:
        """Extract tool calls from model response."""
        tool_calls = []
        
        # Look for tool_call code blocks
        import re
        pattern = r'```tool_call\s*\n(.*?)\n```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                tool_data = json.loads(match.strip())
                tool_call = ToolCall(
                    name=tool_data["name"],
                    arguments=tool_data.get("arguments", {}),
                    call_id=f"call_{len(tool_calls)}"
                )
                tool_calls.append(tool_call)
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Failed to parse tool call: {e}")
                continue
        
        return tool_calls
    
    async def _execute_tool_call(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result."""
        try:
            tool = self.tool_registry.get_tool(tool_call.name)
            if not tool:
                return f"❌ Tool '{tool_call.name}' not found"
            
            result = await tool.execute(**tool_call.arguments)
            return str(result)
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}\n{traceback.format_exc()}")
            return f"❌ Tool execution failed: {str(e)}"
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score for response."""
        # Simple heuristic - can be made more sophisticated
        confidence = 1.0
        
        # Lower confidence for error indicators
        if any(indicator in response.lower() for indicator in ["error", "failed", "unknown", "unsure"]):
            confidence *= 0.7
        
        # Lower confidence for very short responses
        if len(response) < 50:
            confidence *= 0.8
            
        return max(0.1, confidence)
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.logger.info("🔄 Conversation history reset")
    
    async def execute_code(self, code: str, language: str = "python") -> str:
        """Direct code execution interface."""
        return await self.code_executor.execute(code, language)
    
    def add_tool(self, tool: "BaseTool"):
        """Add a custom tool to the agent."""
        self.tool_registry.register(tool)
        self.logger.info(f"🔧 Added tool: {tool.name}")


# Import required components
from .code_executor import CodeExecutor
from .tools import ToolRegistry, BaseTool
from .planner import TaskPlanner
