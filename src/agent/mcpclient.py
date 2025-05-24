import asyncio
from functools import partial
from typing import Any

from langchain_core.tools import BaseTool, tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from protocol.mcpclient import McpClientLike

from .mcp_config import McpServerConfig, load_mcp_configs


class McpClient:
    def __init__(
        self,
        server_config: McpServerConfig,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        """
        Initialize an MCP client.

        Args:
            server_config: The server configuration object from load_mcp_configs.
            command: Override command from config. If None, uses config value.
            args: Override args from config. If None, uses config value.
            env: Override env from config. If None, uses config value.
        """
        # Allow override of individual parameters
        cmd = command if command is not None else server_config.command
        arguments = args if args is not None else server_config.args
        environment = env if env is not None else server_config.env

        # Store the system flag from server_config
        self.is_system_server = server_config.system

        # Create StdioServerParameters without system attribute
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

    async def list_tools(self) -> list[BaseTool]:
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

    def is_system(self) -> bool:
        """Check if the server is a system server"""
        return self.is_system_server

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
        self.clients: dict[str, McpClientLike] = {}
        self.servers = load_mcp_configs(config_path)
        self.all_tools: list[BaseTool] = []

    async def initialize_clients(self) -> dict[str, McpClientLike]:
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
                server_config=server_config,
            )
            await client.connect()
            self.clients[server_name] = client

        return self.clients

    async def get_all_tools(self) -> list[BaseTool]:
        """
        Get all non-system tools from connected MCP clients.

        Returns:
            Combined list of non-system tools from all servers.
        """
        # Clear existing tools list
        self.all_tools = []

        # Collect tools from each client that is NOT marked as system
        for server_name, client in self.clients.items():
            try:
                # Only include non-system servers
                if client.is_system():
                    continue

                tools = await client.list_tools()
                # Could add server name as prefix to tools for disambiguation if needed
                self.all_tools.extend(tools)
            except Exception as e:
                print(f"Error getting tools from server {server_name}: {e}")

        return self.all_tools

    async def get_system_tools(self) -> list[BaseTool]:
        """
        Get tools only from MCP clients marked as system servers.

        Returns:
            List of tools from system servers.
        """
        system_tools = []

        # Collect tools from each client that is marked as system
        for server_name, client in self.clients.items():
            try:
                if not client.is_system():
                    continue

                tools = await client.list_tools()
                system_tools.extend(tools)
            except Exception as e:
                print(f"Error getting tools from system server {server_name}: {e}")

        return system_tools

    async def get_client(self, server_name: str) -> McpClientLike:
        """
        Get a specific MCP client by server name.

        Args:
            server_name: The name of the server to get the client for.

        Returns:
            The connected McpClient instance for the specified server.

        Raises:
            KeyError: If the server name is not found in the clients dictionary.
        """
        if server_name not in self.clients:
            if server_name in self.servers:
                # Create and connect the client if it exists in config but hasn't been initialized
                client = McpClient(
                    server_config=self.servers[server_name],
                )
                await client.connect()
                self.clients[server_name] = client
            else:
                raise KeyError(f"Server '{server_name}' not found in configuration")

        return self.clients[server_name]

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
    # Initialize the manager using async context manager
    async with McpClientManager() as manager:
        # Get all non-system tools (default behavior now)
        non_system_tools = await manager.get_all_tools()
        print(f"Found {len(non_system_tools)} non-system tools")

        # Get only system tools
        system_tools = await manager.get_system_tools()
        print(f"Found {len(system_tools)} system tools")

        # Get all tools including system tools
        all_tools = await manager.get_all_tools_including_system()
        print(f"Found {len(all_tools)} total tools (system + non-system)")

        # Get tools by server
        server_names = list(manager.clients.keys())
        if server_names:
            # Get the first server client (just as an example)
            server_name = server_names[0]
            client = await manager.get_client(server_name)

            # List tools from that specific server
            server_tools = await client.list_tools()
            print(f"\nTools from {server_name}:")
            for tool in server_tools:
                print(f"- {tool}")

            # Check if this is a system server
            is_system = client.is_system()
            print(f"Is {server_name} a system server? {is_system}")

            # Example of calling a tool on a specific server
            try:
                result = await client.call_tool(
                    "read_godoc", arguments={"package_url": "html"}
                )
                print(f"\nTool call result: {result}")
            except Exception as e:
                print(f"\nError calling tool: {e}")


if __name__ == "__main__":
    asyncio.run(main())
