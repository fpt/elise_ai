import logging
import logging.config

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        "simple": {"format": "%(levelname)s: %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "level": "INFO",
            "filename": "logs/app.log",
        },
    },
    "root": {"handlers": ["console", "file"], "level": "INFO"},
}


def setup_logging(debug=False):
    config = LOG_CONFIG.copy()
    if debug:
        # Update only file handler to DEBUG level
        config["handlers"]["file"]["level"] = "DEBUG"
        # Root logger still needs to be at DEBUG to allow DEBUG messages to reach handlers
        config["root"]["level"] = "DEBUG"
        # Console handler stays at INFO

    logging.config.dictConfig(config)
