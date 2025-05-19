import asyncio
from functools import partial
from typing import Any, Dict, List

from langchain_core.tools import BaseTool, tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from protocol.mcpclient import McpClientLike

from .mcp_config import get_default_mcp_server, load_mcp_configs


class McpClient:
    def __init__(
        self,
        config_path: str | None = None,
        server_name: str | None = None,
        command: str | None = None,
        args: List[str] | None = None,
        env: Dict[str, str] | None = None,
    ):
        """
        Initialize an MCP client.

        Args:
            config_path: Path to the mcp.json config file. If None, uses default search path.
            server_name: Name of the server to use from the config. If None, uses the first server.
            command: Override command from config. If None, uses config value.
            args: Override args from config. If None, uses config value.
            env: Override env from config. If None, uses config value.
        """
        # Load configuration
        if server_name:
            servers = load_mcp_configs(config_path)
            if server_name in servers:
                server_config = servers[server_name]
            else:
                server_config = get_default_mcp_server()
        else:
            server_config = get_default_mcp_server()

        # Allow override of individual parameters
        cmd = command if command is not None else server_config.command
        arguments = args if args is not None else server_config.args
        environment = env if env is not None else server_config.env

        self.server_params = StdioServerParameters(
            command=cmd, args=arguments, env=environment
        )
        self.session = None
        self._client_ctx = None
        self.read = None
        self.write = None

    async def connect(self):
        """Initialize connection to the MCP server"""
        # Store the context manager itself
        self._client_ctx = stdio_client(self.server_params)
        # Enter the context
        self.read, self.write = await self._client_ctx.__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        await self.session.initialize()
        return self

    async def close(self):
        """Close connection to the MCP server"""
        if self.session:
            await self.session.__aexit__(None, None, None)
            self.session = None

        if self._client_ctx:
            # Use the stored context manager to exit properly
            await self._client_ctx.__aexit__(None, None, None)
            self._client_ctx = None
            self.read = None
            self.write = None

    async def __aenter__(self):
        """Support for async context manager"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support for async context manager"""
        await self.close()

    async def list_tools(self) -> List[BaseTool]:
        """List available tools"""
        if not self.session:
            await self.connect()

        # Make sure we have a session
        if not self.session:
            raise RuntimeError("Failed to establish connection to MCP server")

        tools = await self.session.list_tools()
        lc_tools = []
        for t in tools.tools:
            tool_name: str = t.name
            description: str = t.description
            inputSchema: dict[str, Any] = t.inputSchema

            # Using the correct form of the tool function
            tool_fn = tool(
                name_or_callable=tool_name,
                runnable=partial(self.call_tool, tool_name),
                description=description,
                args_schema=inputSchema,
            )
            lc_tools.append(tool_fn)
        return lc_tools

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a tool with arguments"""
        if not self.session:
            await self.connect()

        # Make sure we have a session
        if not self.session:
            raise RuntimeError("Failed to establish connection to MCP server")

        return await self.session.call_tool(tool_name, arguments=kwargs)


class McpClientManager:
    """
    Manager for multiple Model Context Protocol (MCP) clients.

    This class loads all MCP server configurations from the specified config file,
    and creates a separate McpClient instance for each server. It provides methods to:
    - Initialize all clients simultaneously
    - Get all tools from all servers combined
    - Access individual clients by server name
    - Close all connections

    Example usage:
    ```python
    # Initialize manager with all servers in config
    async with McpClientManager(config_path="mcp.json") as manager:
        # Get all tools from all servers
        tools = await manager.get_all_tools()

        # Get a specific client
        client = await manager.get_client("server-name")

        # Call a tool on that specific client
        result = await client.call_tool("tool_name", arguments={"arg": "value"})
    # All clients automatically closed when exiting context
    ```
    """

    def __init__(self, config_path: str | None = None):
        """
        Initialize an MCP client manager that handles multiple MCP servers.

        Args:
            config_path: Path to the mcp.json config file. If None, uses default search path.
        """
        self.config_path = config_path
        self.clients: Dict[str, McpClientLike] = {}
        self.servers = load_mcp_configs(config_path)
        self.all_tools: List[BaseTool] = []

    async def initialize_clients(self) -> Dict[str, McpClientLike]:
        """
        Initialize MCP clients for all servers defined in the config.

        Returns:
            Dictionary mapping server names to connected McpClient instances.
        """
        # Close any existing clients
        await self.close_all()

        # Create and connect clients for each server
        for server_name, server_config in self.servers.items():
            client = McpClient(
                config_path=self.config_path,
                server_name=server_name,
                command=server_config.command,
                args=server_config.args,
                env=server_config.env,
            )
            await client.connect()
            self.clients[server_name] = client

        return self.clients

    async def get_all_tools(self) -> List[BaseTool]:
        """
        Get all tools from all connected MCP clients.

        Returns:
            Combined list of all tools from all servers.
        """
        # Clear existing tools list
        self.all_tools = []

        # Collect tools from each client
        for server_name, client in self.clients.items():
            try:
                tools = await client.list_tools()
                # Could add server name as prefix to tools for disambiguation if needed
                self.all_tools.extend(tools)
            except Exception as e:
                print(f"Error getting tools from server {server_name}: {e}")

        return self.all_tools

    async def close_all(self):
        """Close all MCP client connections."""
        for server_name, client in list(self.clients.items()):
            try:
                await client.close()
            except Exception as e:
                print(f"Error closing client for server {server_name}: {e}")

        self.clients.clear()

    async def __aenter__(self):
        """Support for async context manager"""
        await self.initialize_clients()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support for async context manager"""
        await self.close_all()


# Example usage
async def main():
    # Using context manager approach (preferred)
    async with McpClient() as client:
        # List available tools
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools:
            print(f"- {tool}")
        print()

        # Call a tool
        result = await client.call_tool("read_godoc", arguments={"package_url": "html"})
        print(f"Tool call result: {result}")

    # Example with McpClientManager
    async with McpClientManager() as manager:
        tools = await manager.get_all_tools()
        print(f"Found {len(tools)} tools across all MCP servers")

        # Get a specific client if needed
        # client = await manager.get_client("go-dev-mcp")
        # result = await client.call_tool("read_godoc", arguments={"package_url": "html"})


if __name__ == "__main__":
    asyncio.run(main())
