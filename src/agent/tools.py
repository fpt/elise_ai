import glob
import logging
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool


@tool
def get_local_datetime() -> str:
    """Return the current datetime as a string."""
    now = datetime.now().astimezone()
    logging.debug(f"[Tool] Current datetime: {now}")
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


# This function is not tool.
def save_memory(memo: str) -> str:
    """Save a memo to datetime named file."""
    Path("./knowledge").mkdir(exist_ok=True)
    now = datetime.now().astimezone()
    filename = now.strftime("%Y-%m-%d %H-%M-%S %Z") + ".txt"

    with open("./knowledge/" + filename, "w") as f:
        f.write(memo)

    logging.debug(f"[Tool] Memo saved to {filename}")
    return filename


@tool
def search_memory(query: str) -> str:
    """Search for a query in all memo files."""
    Path("./knowledge").mkdir(exist_ok=True)
    files = glob.glob("./knowledge/*.txt")
    results = []

    for filename in files:
        with open(filename, "r") as f:
            content = f.read()
            if query in content:
                results.append(content)

    logging.debug(f"[Tool] Search '{query}': {len(results)} files found")
    return "\n".join(results)
