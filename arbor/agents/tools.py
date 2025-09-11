#!/usr/bin/env python3
"""
Tool System for Arbor Agents.

This module provides a comprehensive tool system including:
- Base tool interface
- Built-in tools (code execution, file operations, web search)
- Tool registry and management
- Dynamic tool loading
"""

import asyncio
import os
import json
import aiohttp
import aiofiles
import subprocess
from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

from .code_executor import CodeExecutor, ExecutionResult


class BaseTool(ABC):
    """Base class for all tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"Tool.{name}")
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool arguments."""
        pass
    
    def validate_args(self, **kwargs) -> tuple[bool, str]:
        """Validate tool arguments."""
        return True, ""


class PythonCodeTool(BaseTool):
    """Tool for executing Python code."""
    
    def __init__(self):
        super().__init__(
            name="python_code",
            description="Execute Python code in a secure environment"
        )
        self.executor = CodeExecutor()
    
    async def execute(self, code: str, **kwargs) -> str:
        """Execute Python code."""
        result = await self.executor.execute(code, "python")
        
        if result.success:
            return f"✅ Code executed successfully:\n{result.output}"
        else:
            return f"❌ Code execution failed:\n{result.error}"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }


class BashCommandTool(BaseTool):
    """Tool for executing bash commands."""
    
    def __init__(self):
        super().__init__(
            name="bash_command",
            description="Execute bash commands safely"
        )
        self.executor = CodeExecutor()
    
    async def execute(self, command: str, **kwargs) -> str:
        """Execute bash command."""
        result = await self.executor.execute(command, "bash")
        
        if result.success:
            return f"✅ Command executed:\n{result.output}"
        else:
            return f"❌ Command failed:\n{result.error}"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to execute"
                }
            },
            "required": ["command"]
        }


class FileReadTool(BaseTool):
    """Tool for reading files."""
    
    def __init__(self):
        super().__init__(
            name="read_file",
            description="Read contents of a file"
        )
    
    async def execute(self, file_path: str, max_size: int = 1024*1024, **kwargs) -> str:
        """Read file contents."""
        try:
            # Security check
            if not self._is_safe_path(file_path):
                return "❌ Access denied: unsafe file path"
            
            # Check file size
            if os.path.getsize(file_path) > max_size:
                return f"❌ File too large (max {max_size} bytes)"
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            return f"📖 File content ({len(content)} chars):\n{content}"
            
        except FileNotFoundError:
            return f"❌ File not found: {file_path}"
        except PermissionError:
            return f"❌ Permission denied: {file_path}"
        except Exception as e:
            return f"❌ Error reading file: {str(e)}"
    
    def _is_safe_path(self, path: str) -> bool:
        """Check if file path is safe to access."""
        # Resolve path
        resolved = os.path.abspath(path)
        
        # Block access to system directories
        blocked_dirs = ['/etc', '/usr', '/bin', '/sbin', '/root', '/sys', '/proc']
        for blocked in blocked_dirs:
            if resolved.startswith(blocked):
                return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "max_size": {
                    "type": "integer",
                    "description": "Maximum file size in bytes",
                    "default": 1048576
                }
            },
            "required": ["file_path"]
        }


class FileWriteTool(BaseTool):
    """Tool for writing files."""
    
    def __init__(self):
        super().__init__(
            name="write_file",
            description="Write content to a file"
        )
    
    async def execute(self, file_path: str, content: str, mode: str = "w", **kwargs) -> str:
        """Write content to file."""
        try:
            # Security check
            if not self._is_safe_path(file_path):
                return "❌ Access denied: unsafe file path"
            
            # Create directory if needed
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            async with aiofiles.open(file_path, mode, encoding='utf-8') as f:
                await f.write(content)
            
            return f"✅ File written: {file_path} ({len(content)} chars)"
            
        except PermissionError:
            return f"❌ Permission denied: {file_path}"
        except Exception as e:
            return f"❌ Error writing file: {str(e)}"
    
    def _is_safe_path(self, path: str) -> bool:
        """Check if file path is safe to write."""
        # Resolve path
        resolved = os.path.abspath(path)
        
        # Block access to system directories
        blocked_dirs = ['/etc', '/usr', '/bin', '/sbin', '/root', '/sys', '/proc']
        for blocked in blocked_dirs:
            if resolved.startswith(blocked):
                return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to write the file"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "mode": {
                    "type": "string",
                    "description": "Write mode (w, a, etc.)",
                    "default": "w"
                }
            },
            "required": ["file_path", "content"]
        }


class WebSearchTool(BaseTool):
    """Tool for web search."""
    
    def __init__(self, api_key: str = None):
        super().__init__(
            name="web_search",
            description="Search the web for information"
        )
        self.api_key = api_key or os.getenv("SEARCH_API_KEY")
    
    async def execute(self, query: str, num_results: int = 5, **kwargs) -> str:
        """Search the web."""
        if not self.api_key:
            return "❌ Web search requires API key (set SEARCH_API_KEY)"
        
        try:
            # Using DuckDuckGo instant answer API (free)
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
            
            # Format results
            results = []
            
            # Instant answer
            if data.get("Abstract"):
                results.append(f"📋 Summary: {data['Abstract']}")
            
            # Related topics
            if data.get("RelatedTopics"):
                for topic in data["RelatedTopics"][:num_results]:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append(f"🔗 {topic['Text']}")
            
            if results:
                return "🔍 Search results:\n" + "\n\n".join(results)
            else:
                return "❌ No search results found"
                
        except Exception as e:
            return f"❌ Search failed: {str(e)}"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }


class CalculatorTool(BaseTool):
    """Tool for mathematical calculations."""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations"
        )
    
    async def execute(self, expression: str, **kwargs) -> str:
        """Evaluate mathematical expression."""
        try:
            # Safe evaluation using restricted environment
            allowed_names = {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow, "len": len,
                "int": int, "float": float, "bool": bool, "str": str
            }
            
            # Import math functions
            import math
            for name in dir(math):
                if not name.startswith("_"):
                    allowed_names[name] = getattr(math, name)
            
            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            return f"🧮 Calculation result: {expression} = {result}"
            
        except Exception as e:
            return f"❌ Calculation failed: {str(e)}"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }


class GitTool(BaseTool):
    """Tool for Git operations."""
    
    def __init__(self):
        super().__init__(
            name="git",
            description="Perform Git operations"
        )
    
    async def execute(self, command: str, repo_path: str = ".", **kwargs) -> str:
        """Execute git command."""
        try:
            # Safety check
            allowed_commands = [
                "status", "log", "diff", "show", "branch",
                "remote", "fetch", "pull", "add", "commit", "push"
            ]
            
            cmd_parts = command.split()
            if not cmd_parts or cmd_parts[0] not in allowed_commands:
                return f"❌ Git command not allowed: {command}"
            
            # Execute git command
            result = subprocess.run(
                ["git"] + cmd_parts,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return f"✅ Git command successful:\n{result.stdout}"
            else:
                return f"❌ Git command failed:\n{result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "❌ Git command timed out"
        except Exception as e:
            return f"❌ Git command failed: {str(e)}"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Git command to execute (without 'git' prefix)"
                },
                "repo_path": {
                    "type": "string",
                    "description": "Path to git repository",
                    "default": "."
                }
            },
            "required": ["command"]
        }


class ToolRegistry:
    """Registry for managing tools."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.logger = logging.getLogger(f"{__class__.__name__}")
    
    def register(self, tool: BaseTool):
        """Register a new tool."""
        self.tools[tool.name] = tool
        self.logger.info(f"🔧 Registered tool: {tool.name}")
    
    def unregister(self, name: str):
        """Unregister a tool."""
        if name in self.tools:
            del self.tools[name]
            self.logger.info(f"🗑️ Unregistered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    def get_tools_description(self) -> str:
        """Get description of all tools."""
        descriptions = []
        for tool in self.tools.values():
            schema = tool.get_schema()
            descriptions.append(f"• {tool.name}: {tool.description}")
            
        return "\n".join(descriptions)
    
    def get_tools_schema(self) -> Dict[str, Any]:
        """Get JSON schema for all tools."""
        schema = {}
        for name, tool in self.tools.items():
            schema[name] = tool.get_schema()
        return schema
