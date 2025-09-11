#!/usr/bin/env python3
"""
Agentic Inference Script for Arbor.

This script provides a complete agentic interface for Arbor with:
- Interactive chat with tool calling
- Code execution capabilities
- MCP integration
- Multi-step reasoning
- Task planning and execution
"""

import asyncio
import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from arbor.agents import ArborAgent, MCPClient, create_mcp_bridge
from arbor.agents.tools import (
    PythonCodeTool, BashCommandTool, FileReadTool, 
    FileWriteTool, WebSearchTool, CalculatorTool, GitTool
)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('arbor_agent.log')
        ]
    )


async def interactive_chat(agent: ArborAgent):
    """Interactive chat interface with the agent."""
    print("🌳 Arbor Agentic Interface")
    print("==========================")
    print("Type 'exit' to quit, 'reset' to clear conversation, 'help' for commands")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() == 'exit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'reset':
                agent.reset_conversation()
                print("🔄 Conversation reset")
                continue
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'tools':
                print_available_tools(agent)
                continue
            elif user_input.lower().startswith('save '):
                filename = user_input[5:].strip()
                save_conversation(agent, filename)
                continue
            
            # Process user message
            print("🤖 Arbor: ", end="", flush=True)
            
            response = await agent.chat(
                user_input,
                max_iterations=10,
                temperature=0.7
            )
            
            print(response.content)
            
            # Show reasoning if available
            if response.reasoning:
                print(f"\n🧠 Reasoning:\n{response.reasoning}")
            
            # Show metadata
            if response.metadata:
                iterations = response.metadata.get('iterations', 0)
                if iterations > 1:
                    print(f"\n📊 Completed in {iterations} iterations")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logging.error(f"Chat error: {e}")


def print_help():
    """Print help information."""
    help_text = """
🌳 Arbor Agentic Interface Commands:

Basic Commands:
  help     - Show this help message
  exit     - Exit the application
  reset    - Reset conversation history
  tools    - Show available tools
  save <filename> - Save conversation to file

Example Interactions:
  "Calculate the factorial of 10"
  "Read the file config.yaml and analyze its structure"
  "Write a Python script to sort a list of numbers"
  "Search for information about quantum computing"
  "Create a plan to build a web scraper"

Features:
  ✅ Code execution (Python, Bash)
  ✅ File operations (read, write)
  ✅ Web search capabilities
  ✅ Mathematical calculations
  ✅ Multi-step reasoning
  ✅ Tool calling and chaining
  ✅ Adaptive context windows
"""
    print(help_text)


def print_available_tools(agent: ArborAgent):
    """Print available tools."""
    tools = agent.tool_registry.list_tools()
    print(f"\n🔧 Available Tools ({len(tools)}):")
    print("=" * 30)
    
    for tool_name in tools:
        tool = agent.tool_registry.get_tool(tool_name)
        print(f"• {tool_name}: {tool.description}")
    
    print()


def save_conversation(agent: ArborAgent, filename: str):
    """Save conversation history to file."""
    try:
        conversation_data = {
            "conversation_history": agent.conversation_history,
            "timestamp": str(asyncio.get_event_loop().time())
        }
        
        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        print(f"💾 Conversation saved to {filename}")
        
    except Exception as e:
        print(f"❌ Failed to save conversation: {e}")


async def run_batch_commands(agent: ArborAgent, commands_file: str):
    """Run commands from a file."""
    try:
        with open(commands_file, 'r') as f:
            commands = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"🚀 Running {len(commands)} commands from {commands_file}")
        
        for i, command in enumerate(commands, 1):
            print(f"\n📝 Command {i}/{len(commands)}: {command}")
            print("-" * 50)
            
            response = await agent.chat(command, max_iterations=10)
            print(f"🤖 Response: {response.content}")
            
            if response.reasoning:
                print(f"🧠 Reasoning: {response.reasoning}")
        
        print(f"\n✅ Completed all {len(commands)} commands")
        
    except FileNotFoundError:
        print(f"❌ Commands file not found: {commands_file}")
    except Exception as e:
        print(f"❌ Error running batch commands: {e}")


async def setup_mcp_integration(agent: ArborAgent, mcp_urls: List[str]):
    """Setup MCP integration."""
    if not mcp_urls:
        return []
    
    print(f"🔗 Connecting to {len(mcp_urls)} MCP servers...")
    
    try:
        clients = await create_mcp_bridge(agent, mcp_urls)
        print(f"✅ Connected to {len(clients)} MCP servers")
        
        # List added tools
        for client in clients:
            tools = client.get_available_tools()
            if tools:
                print(f"  📋 {client.server_url}: {', '.join(tools)}")
        
        return clients
        
    except Exception as e:
        print(f"❌ MCP integration failed: {e}")
        logging.error(f"MCP setup error: {e}")
        return []


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Arbor Agentic Inference Interface")
    parser.add_argument(
        "--model", 
        default="Noema-Research/arbor-base",
        help="Model name or path"
    )
    parser.add_argument(
        "--multimodal", 
        action="store_true",
        help="Use multimodal model"
    )
    parser.add_argument(
        "--batch", 
        help="Run commands from file instead of interactive mode"
    )
    parser.add_argument(
        "--mcp-servers", 
        nargs="*",
        help="MCP server URLs to connect to"
    )
    parser.add_argument(
        "--max-context", 
        type=int, 
        default=131072,
        help="Maximum context length"
    )
    parser.add_argument(
        "--device", 
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--add-tools", 
        nargs="*",
        help="Additional tools to add (git, etc.)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    print("🌳 Initializing Arbor Agent...")
    
    # Create agent
    agent = ArborAgent(
        model_name_or_path=args.model,
        multimodal=args.multimodal,
        max_context=args.max_context,
        device=args.device
    )
    
    # Add additional tools
    if args.add_tools:
        if "git" in args.add_tools:
            agent.add_tool(GitTool())
            print("🔧 Added Git tool")
    
    # Setup MCP integration
    mcp_clients = []
    if args.mcp_servers:
        mcp_clients = await setup_mcp_integration(agent, args.mcp_servers)
    
    try:
        # Run in batch or interactive mode
        if args.batch:
            await run_batch_commands(agent, args.batch)
        else:
            await interactive_chat(agent)
    
    finally:
        # Cleanup MCP connections
        for client in mcp_clients:
            try:
                await client.disconnect()
            except Exception as e:
                logging.warning(f"Error disconnecting MCP client: {e}")


if __name__ == "__main__":
    asyncio.run(main())
