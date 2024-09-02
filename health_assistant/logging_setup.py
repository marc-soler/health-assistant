import logging
import sys
from logging.config import dictConfig


class LoggerSetup:
    """
    LoggerSetup is responsible for configuring the logging for the application.
    It ensures that logs are formatted correctly and output to the appropriate streams.
    """

    def __init__(self):
        """
        Initializes the LoggerSetup instance.
        No arguments are required for initialization.
        """
        self.logger = None

    def setup_logging(self):
        """
        Configures the logging settings for the application.
        - Logs are formatted in JSON for better integration with logging systems.
        - Logs are sent to stdout (for standard logs) and stderr (for errors).
        """
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)s %(message)s",
                },
                "json": {
                    "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                    "formatter": "json",  # Change to 'default' if you prefer plain text logs
                },
                "error_console": {
                    "class": "logging.StreamHandler",
                    "stream": sys.stderr,
                    "formatter": "json",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["console", "error_console"],
            },
            "loggers": {
                "__main__": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }

        dictConfig(logging_config)
        self.logger = logging.getLogger(__name__)

    def get_logger(self):
        """
        Returns the logger instance configured by LoggerSetup.
        If the logger has not been configured yet, it calls setup_logging first.

        Returns:
            logging.Logger: The configured logger instance.
        """
        if self.logger is None:
            self.setup_logging()
        return self.logger
