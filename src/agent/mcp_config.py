import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class McpServerConfig:
    name: str
    type: str
    command: str
    args: List[str]
    env: Dict[str, str] | None
    system: bool = False

    @classmethod
    def from_dict(cls, name: str, config_dict: dict):
        return cls(
            name=name,
            type=config_dict.get("type", "stdio"),
            command=config_dict.get("command", ""),
            args=config_dict.get("args", []),
            env=config_dict.get("env", {}),
            system=config_dict.get("system", False),
        )


def load_mcp_configs(config_path: Optional[str] = None) -> Dict[str, McpServerConfig]:
    """
    Load MCP server configurations from a JSON file.

    Args:
        config_path: Path to the configuration file. If None, searches for mcp.json
                     in the current directory and parent directories.

    Returns:
        Dictionary mapping server names to McpServerConfig objects.
    """
    if config_path is None:
        # Search for mcp.json in current directory and parent directories
        current_dir = os.path.abspath(os.path.dirname(__file__))
        while True:
            config_path = os.path.join(current_dir, "mcp.json")
            if os.path.exists(config_path):
                break

            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached root directory
                # Default to project root if not found
                config_path = os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
                    "mcp.json",
                )
                break

            current_dir = parent_dir

    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)

        servers_config = config_data.get("mcp", {}).get("servers", {})
        if not servers_config:
            # Fallback to default configuration if no servers defined
            return {
                "default": McpServerConfig(
                    name="default",
                    type="stdio",
                    command="godevmcp",
                    args=["serve"],
                    env={},
                    system=False,
                )
            }

        return {
            name: McpServerConfig.from_dict(name, config)
            for name, config in servers_config.items()
        }
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default configuration if file not found or invalid
        return {
            "default": McpServerConfig(
                name="default", type="stdio", command="godevmcp", args=["serve"], env={}
            )
        }
