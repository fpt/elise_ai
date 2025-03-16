import logging
from datetime import datetime

from langchain_core.tools import tool


@tool
def get_local_datetime() -> str:
    """Return the current datetime as a string."""
    now = datetime.now().astimezone()
    logging.debug(f"[Tool] Current datetime: {now}")
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")
