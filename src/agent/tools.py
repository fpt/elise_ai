import glob
import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool


@tool
def get_local_datetime() -> str:
    """Return the current datetime as a string."""
    now = datetime.now().astimezone()
    logging.debug(f"[Tool] Current datetime: {now}")
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


@tool
def get_cwd() -> str:
    """Return the current working directory."""
    cwd = Path.cwd()
    logging.debug(f"[Tool] Current working directory: {cwd}")
    return str(cwd)


@tool
def remind_memory(
    keyword: Annotated[
        str, "The keywords to search for in memory files (space-separated)."
    ],
) -> str:
    """Remind memory by searching for space-separated keywords in memory files (case-insensitive)."""
    Path("./knowledge").mkdir(exist_ok=True)
    files = glob.glob("./knowledge/*.txt")
    results = []

    # Split the input by whitespace to get individual keywords
    keywords = keyword.lower().split()

    for filename in files:
        with open(filename, "r") as f:
            content = f.read()
            content_lower = content.lower()

            # Check if all keywords are in the content (case-insensitive)
            if all(kw in content_lower for kw in keywords):
                results.append(content)

    logging.debug(f"[Tool] Search '{keyword}': {len(results)} files found")
    return "\n".join(results)


# This function is not tool.
def save_memory(memory: str) -> str:
    """Save a memory to datetime named file."""
    Path("./knowledge").mkdir(exist_ok=True)
    now = datetime.now().astimezone()
    filename = now.strftime("%Y-%m-%d %H-%M-%S %Z") + ".txt"

    with open("./knowledge/" + filename, "w") as f:
        f.write(memory)

    logging.debug(f"[Tool] Memory saved to {filename}")
    return filename
