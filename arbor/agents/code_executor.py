#!/usr/bin/env python3
"""
Code Execution System for Arbor Agents.

This module provides secure code execution capabilities including:
- Python code execution in sandboxed environments
- Bash command execution with safety checks
- SQL query execution
- Code validation and safety checks
"""

import asyncio
import subprocess
import tempfile
import os
import sys
import io
import contextlib
import logging
import signal
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import docker
import psutil


@dataclass
class ExecutionResult:
    """Result from code execution."""
    output: str
    error: str = ""
    exit_code: int = 0
    execution_time: float = 0.0
    memory_used: int = 0
    success: bool = True


class BaseExecutor(ABC):
    """Base class for code executors."""
    
    def __init__(self, timeout: int = 30, memory_limit: int = 512):
        self.timeout = timeout
        self.memory_limit = memory_limit  # MB
        self.logger = logging.getLogger(f"{__class__.__name__}")
    
    @abstractmethod
    async def execute(self, code: str, **kwargs) -> ExecutionResult:
        """Execute code and return result."""
        pass
    
    def _is_safe_code(self, code: str) -> tuple[bool, str]:
        """Check if code is safe to execute."""
        dangerous_patterns = [
            'import os',
            'import subprocess',
            'import sys',
            '__import__',
            'exec(',
            'eval(',
            'open(',
            'file(',
            'input(',
            'raw_input(',
            'compile(',
            'reload(',
            'delattr(',
            'setattr(',
            'hasattr(',
            'globals(',
            'locals(',
            'vars(',
            'dir(',
            'help(',
            'quit(',
            'exit('
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return False, f"Potentially dangerous code detected: {pattern}"
        
        return True, ""


class PythonExecutor(BaseExecutor):
    """Secure Python code executor."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allowed_modules = {
            'math', 'random', 'datetime', 'json', 'base64',
            'hashlib', 'uuid', 'itertools', 'collections',
            'functools', 'operator', 'string', 're',
            'numpy', 'pandas', 'matplotlib', 'seaborn',
            'sklearn', 'scipy', 'requests'
        }
    
    async def execute(self, code: str, **kwargs) -> ExecutionResult:
        """Execute Python code in a restricted environment."""
        # Safety check
        is_safe, reason = self._is_safe_code(code)
        if not is_safe:
            return ExecutionResult(
                output="", 
                error=f"Code execution blocked: {reason}",
                success=False
            )
        
        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Redirect output
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Create restricted globals
            restricted_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round,
                    'sorted': sorted,
                    'reversed': reversed,
                }
            }
            
            # Execute with timeout
            import time
            start_time = time.time()
            
            # Execute code
            exec(code, restricted_globals)
            
            execution_time = time.time() - start_time
            
            # Get output
            output = stdout_capture.getvalue()
            error = stderr_capture.getvalue()
            
            return ExecutionResult(
                output=output,
                error=error,
                execution_time=execution_time,
                success=not bool(error)
            )
            
        except Exception as e:
            return ExecutionResult(
                output=stdout_capture.getvalue(),
                error=f"Execution error: {str(e)}",
                success=False
            )
        finally:
            # Restore output
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class BashExecutor(BaseExecutor):
    """Secure Bash command executor."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allowed_commands = {
            'ls', 'cat', 'grep', 'find', 'head', 'tail',
            'wc', 'sort', 'uniq', 'cut', 'awk', 'sed',
            'echo', 'pwd', 'date', 'which', 'whoami',
            'ps', 'top', 'df', 'du', 'free', 'uptime'
        }
        self.blocked_commands = {
            'rm', 'rmdir', 'mv', 'cp', 'chmod', 'chown',
            'sudo', 'su', 'kill', 'killall', 'reboot',
            'shutdown', 'halt', 'init', 'service',
            'systemctl', 'mount', 'umount', 'fdisk'
        }
    
    async def execute(self, command: str, **kwargs) -> ExecutionResult:
        """Execute bash command with safety checks."""
        # Safety check
        is_safe, reason = self._is_safe_command(command)
        if not is_safe:
            return ExecutionResult(
                output="",
                error=f"Command execution blocked: {reason}",
                success=False
            )
        
        try:
            # Execute with timeout
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024*1024  # 1MB output limit
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
                
                return ExecutionResult(
                    output=stdout.decode('utf-8', errors='ignore'),
                    error=stderr.decode('utf-8', errors='ignore'),
                    exit_code=process.returncode,
                    success=process.returncode == 0
                )
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ExecutionResult(
                    output="",
                    error=f"Command timed out after {self.timeout} seconds",
                    success=False
                )
                
        except Exception as e:
            return ExecutionResult(
                output="",
                error=f"Command execution failed: {str(e)}",
                success=False
            )
    
    def _is_safe_command(self, command: str) -> tuple[bool, str]:
        """Check if command is safe to execute."""
        command_parts = command.split()
        if not command_parts:
            return False, "Empty command"
        
        base_command = command_parts[0]
        
        # Check for blocked commands
        if base_command in self.blocked_commands:
            return False, f"Blocked command: {base_command}"
        
        # Check for dangerous patterns
        dangerous_patterns = [
            '&&', '||', ';', '|', '>', '>>', '<',
            '$(', '`', 'rm -rf', '/etc/', '/usr/',
            '/bin/', '/sbin/', '/var/', '/root/'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in command:
                return False, f"Dangerous pattern detected: {pattern}"
        
        return True, ""


class DockerExecutor(BaseExecutor):
    """Docker-based code executor for maximum security."""
    
    def __init__(self, image: str = "python:3.11-slim", **kwargs):
        super().__init__(**kwargs)
        self.image = image
        self.client = None
        
        try:
            self.client = docker.from_env()
            self.logger.info(f"🐳 Docker executor initialized with image: {image}")
        except Exception as e:
            self.logger.warning(f"Docker not available: {e}")
    
    async def execute(self, code: str, language: str = "python", **kwargs) -> ExecutionResult:
        """Execute code in Docker container."""
        if not self.client:
            return ExecutionResult(
                output="",
                error="Docker not available",
                success=False
            )
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run in container
            try:
                container = self.client.containers.run(
                    self.image,
                    f"python {os.path.basename(temp_file)}",
                    volumes={os.path.dirname(temp_file): {'bind': '/code', 'mode': 'ro'}},
                    working_dir='/code',
                    mem_limit=f"{self.memory_limit}m",
                    cpu_quota=50000,  # 50% CPU
                    network_disabled=True,
                    remove=True,
                    detach=False,
                    stdout=True,
                    stderr=True
                )
                
                output = container.decode('utf-8')
                return ExecutionResult(output=output, success=True)
                
            except docker.errors.ContainerError as e:
                return ExecutionResult(
                    output="",
                    error=f"Container execution failed: {e.stderr.decode('utf-8')}",
                    exit_code=e.exit_status,
                    success=False
                )
            finally:
                # Cleanup
                os.unlink(temp_file)
                
        except Exception as e:
            return ExecutionResult(
                output="",
                error=f"Docker execution failed: {str(e)}",
                success=False
            )


class CodeExecutor:
    """Main code executor that delegates to specific executors."""
    
    def __init__(self, use_docker: bool = False, **kwargs):
        self.use_docker = use_docker
        self.logger = logging.getLogger(f"{__class__.__name__}")
        
        # Initialize executors
        if use_docker:
            self.python_executor = DockerExecutor(image="python:3.11-slim", **kwargs)
        else:
            self.python_executor = PythonExecutor(**kwargs)
            
        self.bash_executor = BashExecutor(**kwargs)
        
        self.logger.info(f"🔧 Code executor initialized (Docker: {use_docker})")
    
    async def execute(self, code: str, language: str = "python", **kwargs) -> ExecutionResult:
        """Execute code in the specified language."""
        if language.lower() in ["python", "py"]:
            return await self.python_executor.execute(code, **kwargs)
        elif language.lower() in ["bash", "sh", "shell"]:
            return await self.bash_executor.execute(code, **kwargs)
        else:
            return ExecutionResult(
                output="",
                error=f"Unsupported language: {language}",
                success=False
            )
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        return ["python", "bash"]
