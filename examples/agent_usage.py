#!/usr/bin/env python3
"""
Example: Arbor Agentic AI Usage

This script demonstrates the comprehensive agentic capabilities of Arbor
including tool calling, code execution, and multi-step reasoning.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from arbor.agents import ArborAgent
from arbor.agents.tools import PythonCodeTool, FileWriteTool, CalculatorTool


async def basic_agent_example():
    """Basic agent usage with tool calling."""
    print("🌳 Basic Arbor Agent Example")
    print("=" * 40)
    
    # Initialize agent
    agent = ArborAgent(
        model_name_or_path="Noema-Research/arbor-base",
        max_context=32768
    )
    
    # Example 1: Mathematical calculation
    print("\n📊 Example 1: Mathematical Calculation")
    response = await agent.chat(
        "Calculate the compound interest for $10,000 invested at 5% annually for 10 years"
    )
    print(f"Response: {response.content}")
    
    # Example 2: Code generation and execution  
    print("\n💻 Example 2: Code Generation")
    response = await agent.chat(
        "Write a Python function to find all prime numbers up to 100 and execute it"
    )
    print(f"Response: {response.content}")
    
    # Example 3: File operations
    print("\n📁 Example 3: File Operations")
    response = await agent.chat(
        "Create a CSV file with sample sales data for 5 products and their monthly sales"
    )
    print(f"Response: {response.content}")


async def advanced_reasoning_example():
    """Advanced multi-step reasoning example."""
    print("\n🧠 Advanced Multi-Step Reasoning Example")
    print("=" * 50)
    
    agent = ArborAgent(
        model_name_or_path="Noema-Research/arbor-base",
        max_context=65536
    )
    
    complex_task = """
    I need to analyze the performance of a hypothetical e-commerce website.
    Please:
    1. Create sample data for 1000 customers with purchases
    2. Calculate key metrics (total revenue, average order value, etc.)
    3. Identify top customers and products
    4. Generate a summary report with insights
    5. Save all results to files
    """
    
    print(f"🎯 Complex Task: {complex_task}")
    print("\n🚀 Executing multi-step plan...")
    
    response = await agent.chat(complex_task, max_iterations=15)
    
    print(f"\n✅ Final Result: {response.content}")
    
    if response.reasoning:
        print(f"\n🧠 Reasoning Chain:\n{response.reasoning}")
    
    if response.metadata:
        iterations = response.metadata.get('iterations', 0)
        print(f"\n📊 Completed in {iterations} iterations")


async def code_execution_demo():
    """Demonstrate code execution capabilities."""
    print("\n🐍 Code Execution Demo")
    print("=" * 30)
    
    agent = ArborAgent(model_name_or_path="Noema-Research/arbor-base")
    
    # Python code execution
    code_tasks = [
        "Create a function to calculate Fibonacci sequence up to n=20",
        "Generate a random dataset and create a simple visualization",
        "Implement a basic sorting algorithm and test it with sample data",
        "Create a simple web scraper function (demonstration only)"
    ]
    
    for i, task in enumerate(code_tasks, 1):
        print(f"\n📝 Task {i}: {task}")
        response = await agent.chat(task)
        print(f"✅ Result: {response.content[:200]}...")


async def tool_calling_showcase():
    """Showcase various tool calling capabilities."""
    print("\n🔧 Tool Calling Showcase")
    print("=" * 35)
    
    agent = ArborAgent(model_name_or_path="Noema-Research/arbor-base")
    
    # Show available tools
    tools = agent.tool_registry.list_tools()
    print(f"🛠️ Available tools: {', '.join(tools)}")
    
    # Tool usage examples
    tool_examples = [
        "Calculate the area of a circle with radius 5.7 meters",
        "Create a text file with today's date and a welcome message",
        "Generate a list of prime numbers between 50 and 100",
        "Search for information about machine learning trends in 2024"
    ]
    
    for example in tool_examples:
        print(f"\n🎯 Task: {example}")
        response = await agent.chat(example)
        print(f"🤖 Response: {response.content[:150]}...")


async def conversation_demo():
    """Demonstrate conversational memory and context."""
    print("\n💬 Conversation Demo")
    print("=" * 25)
    
    agent = ArborAgent(model_name_or_path="Noema-Research/arbor-base")
    
    conversation_flow = [
        "I'm working on a data analysis project for sales data",
        "Can you help me create a sample dataset with 100 customers?",
        "Now calculate the total revenue from this dataset",
        "What's the average order value?",
        "Create a simple report summarizing these findings"
    ]
    
    for i, message in enumerate(conversation_flow, 1):
        print(f"\n👤 User {i}: {message}")
        response = await agent.chat(message)
        print(f"🤖 Arbor: {response.content[:200]}...")
    
    print(f"\n📊 Conversation length: {len(agent.conversation_history)} turns")


async def error_handling_demo():
    """Demonstrate error handling and recovery."""
    print("\n⚠️ Error Handling Demo")
    print("=" * 30)
    
    agent = ArborAgent(model_name_or_path="Noema-Research/arbor-base")
    
    # Intentionally problematic requests
    problematic_tasks = [
        "Read a file that doesn't exist: 'nonexistent.txt'",
        "Execute this invalid Python code: 'print(undefined_variable)'",
        "Calculate the square root of -1 using the calculator"
    ]
    
    for task in problematic_tasks:
        print(f"\n🧪 Testing: {task}")
        response = await agent.chat(task)
        print(f"🤖 Response: {response.content[:200]}...")


async def main():
    """Run all examples."""
    print("🌳 Arbor Agentic AI Comprehensive Demo")
    print("=" * 50)
    
    demos = [
        ("Basic Agent Usage", basic_agent_example),
        ("Advanced Reasoning", advanced_reasoning_example),  
        ("Code Execution", code_execution_demo),
        ("Tool Calling", tool_calling_showcase),
        ("Conversation Memory", conversation_demo),
        ("Error Handling", error_handling_demo)
    ]
    
    for name, demo_func in demos:
        try:
            print(f"\n🚀 Running {name}...")
            await demo_func()
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
        
        # Pause between demos
        await asyncio.sleep(1)
    
    print("\n🎉 All demos completed!")
    print("\nTo run individual examples:")
    print("python examples/agent_usage.py")


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(main())
