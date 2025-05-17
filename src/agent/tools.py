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
def remind_memory(
    keyword: Annotated[str, "The keyword to search for in memory files."],
) -> str:
    """Remind memory by searching for a keyword in memory files."""
    Path("./knowledge").mkdir(exist_ok=True)
    files = glob.glob("./knowledge/*.txt")
    results = []

    for filename in files:
        with open(filename, "r") as f:
            content = f.read()
            if keyword in content:
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
