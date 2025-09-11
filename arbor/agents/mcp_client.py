#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Integration for Arbor.

This module provides MCP client and server implementations for:
- Standardized tool interfaces
- Cross-model communication
- Resource sharing and management
- Standardized context handling
"""

import asyncio
import json
import uuid
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import websockets
import aiohttp

from .tools import BaseTool, ToolRegistry


@dataclass
class MCPMessage:
    """MCP protocol message."""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class MCPTool:
    """MCP tool definition."""
    name: str
    description: str
    inputSchema: Dict[str, Any]


@dataclass
class MCPResource:
    """MCP resource definition."""
    uri: str
    name: str
    description: str
    mimeType: str


class MCPClient:
    """MCP client for connecting to MCP servers."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.websocket = None
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.logger = logging.getLogger(f"{__class__.__name__}")
        self._request_id = 0
    
    async def connect(self):
        """Connect to MCP server."""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.logger.info(f"🔗 Connected to MCP server: {self.server_url}")
            
            # Initialize connection
            await self._initialize()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.logger.info("🔌 Disconnected from MCP server")
    
    async def _initialize(self):
        """Initialize MCP connection."""
        # Get server capabilities
        capabilities_response = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {}
            },
            "clientInfo": {
                "name": "Arbor Agent",
                "version": "1.0.0"
            }
        })
        
        # List available tools
        await self._list_tools()
        
        # List available resources
        await self._list_resources()
    
    async def _send_request(self, method: str, params: Dict[str, Any] = None) -> Any:
        """Send MCP request and wait for response."""
        self._request_id += 1
        request = MCPMessage(
            id=str(self._request_id),
            method=method,
            params=params or {}
        )
        
        await self.websocket.send(json.dumps(asdict(request)))
        
        # Wait for response
        response_data = await self.websocket.recv()
        response = json.loads(response_data)
        
        if "error" in response:
            raise Exception(f"MCP error: {response['error']}")
        
        return response.get("result")
    
    async def _list_tools(self):
        """List available tools from server."""
        try:
            result = await self._send_request("tools/list")
            
            for tool_data in result.get("tools", []):
                tool = MCPTool(
                    name=tool_data["name"],
                    description=tool_data["description"],
                    inputSchema=tool_data["inputSchema"]
                )
                self.tools[tool.name] = tool
                
            self.logger.info(f"📋 Listed {len(self.tools)} MCP tools")
            
        except Exception as e:
            self.logger.error(f"Failed to list tools: {e}")
    
    async def _list_resources(self):
        """List available resources from server."""
        try:
            result = await self._send_request("resources/list")
            
            for resource_data in result.get("resources", []):
                resource = MCPResource(
                    uri=resource_data["uri"],
                    name=resource_data["name"],
                    description=resource_data["description"],
                    mimeType=resource_data["mimeType"]
                )
                self.resources[resource.uri] = resource
                
            self.logger.info(f"📁 Listed {len(self.resources)} MCP resources")
            
        except Exception as e:
            self.logger.error(f"Failed to list resources: {e}")
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        
        try:
            result = await self._send_request("tools/call", {
                "name": name,
                "arguments": arguments
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Tool call failed: {e}")
            raise
    
    async def read_resource(self, uri: str) -> str:
        """Read a resource from the MCP server."""
        if uri not in self.resources:
            raise ValueError(f"Resource not found: {uri}")
        
        try:
            result = await self._send_request("resources/read", {
                "uri": uri
            })
            
            return result.get("contents", [{}])[0].get("text", "")
            
        except Exception as e:
            self.logger.error(f"Resource read failed: {e}")
            raise
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
    
    def get_available_resources(self) -> List[str]:
        """Get list of available resource URIs."""
        return list(self.resources.keys())


class MCPServer:
    """MCP server for exposing Arbor capabilities."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.tool_registry = ToolRegistry()
        self.resources: Dict[str, MCPResource] = {}
        self.clients = set()
        self.logger = logging.getLogger(f"{__class__.__name__}")
    
    def add_tool(self, tool: BaseTool):
        """Add a tool to the MCP server."""
        self.tool_registry.register(tool)
        self.logger.info(f"🔧 Added MCP tool: {tool.name}")
    
    def add_resource(self, resource: MCPResource):
        """Add a resource to the MCP server."""
        self.resources[resource.uri] = resource
        self.logger.info(f"📁 Added MCP resource: {resource.uri}")
    
    async def start(self):
        """Start the MCP server."""
        self.logger.info(f"🚀 Starting MCP server on {self.host}:{self.port}")
        
        async def handler(websocket, path):
            self.clients.add(websocket)
            try:
                await self._handle_client(websocket)
            finally:
                self.clients.remove(websocket)
        
        await websockets.serve(handler, self.host, self.port)
        self.logger.info(f"✅ MCP server started on ws://{self.host}:{self.port}")
    
    async def _handle_client(self, websocket):
        """Handle MCP client connection."""
        self.logger.info("🤝 New MCP client connected")
        
        async for message in websocket:
            try:
                data = json.loads(message)
                response = await self._process_request(data)
                
                if response:
                    await websocket.send(json.dumps(asdict(response)))
                    
            except Exception as e:
                self.logger.error(f"Error handling client message: {e}")
                error_response = MCPMessage(
                    id=data.get("id"),
                    error={"code": -32603, "message": str(e)}
                )
                await websocket.send(json.dumps(asdict(error_response)))
    
    async def _process_request(self, data: Dict[str, Any]) -> Optional[MCPMessage]:
        """Process MCP request."""
        method = data.get("method")
        params = data.get("params", {})
        request_id = data.get("id")
        
        if method == "initialize":
            return MCPMessage(
                id=request_id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "resources": {"listChanged": True}
                    },
                    "serverInfo": {
                        "name": "Arbor MCP Server",
                        "version": "1.0.0"
                    }
                }
            )
        
        elif method == "tools/list":
            tools = []
            for tool in self.tool_registry.tools.values():
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.get_schema()
                })
            
            return MCPMessage(
                id=request_id,
                result={"tools": tools}
            )
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                raise ValueError(f"Tool not found: {tool_name}")
            
            result = await tool.execute(**arguments)
            
            return MCPMessage(
                id=request_id,
                result={
                    "content": [{"type": "text", "text": str(result)}]
                }
            )
        
        elif method == "resources/list":
            resources = []
            for resource in self.resources.values():
                resources.append({
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mimeType": resource.mimeType
                })
            
            return MCPMessage(
                id=request_id,
                result={"resources": resources}
            )
        
        elif method == "resources/read":
            uri = params.get("uri")
            resource = self.resources.get(uri)
            
            if not resource:
                raise ValueError(f"Resource not found: {uri}")
            
            # Read resource content (implement based on resource type)
            content = await self._read_resource_content(resource)
            
            return MCPMessage(
                id=request_id,
                result={
                    "contents": [{"type": "text", "text": content}]
                }
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    async def _read_resource_content(self, resource: MCPResource) -> str:
        """Read content of a resource."""
        # Implement based on resource type
        if resource.uri.startswith("file://"):
            file_path = resource.uri[7:]  # Remove "file://" prefix
            try:
                with open(file_path, 'r') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {e}"
        
        return f"Resource content for {resource.uri}"


class MCPToolAdapter(BaseTool):
    """Adapter to use MCP tools as Arbor tools."""
    
    def __init__(self, mcp_client: MCPClient, tool_name: str):
        self.mcp_client = mcp_client
        self.mcp_tool = mcp_client.tools[tool_name]
        
        super().__init__(
            name=self.mcp_tool.name,
            description=self.mcp_tool.description
        )
    
    async def execute(self, **kwargs) -> Any:
        """Execute the MCP tool."""
        return await self.mcp_client.call_tool(self.name, kwargs)
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema."""
        return self.mcp_tool.inputSchema


async def create_mcp_bridge(
    arbor_agent,
    mcp_server_urls: List[str]
) -> List[MCPClient]:
    """
    Create bridge between Arbor agent and MCP servers.
    
    This function connects to multiple MCP servers and adds their
    tools to the Arbor agent.
    """
    clients = []
    
    for url in mcp_server_urls:
        try:
            client = MCPClient(url)
            await client.connect()
            
            # Add MCP tools to Arbor agent
            for tool_name in client.get_available_tools():
                adapter = MCPToolAdapter(client, tool_name)
                arbor_agent.add_tool(adapter)
            
            clients.append(client)
            
        except Exception as e:
            logging.error(f"Failed to connect to MCP server {url}: {e}")
    
    return clients
