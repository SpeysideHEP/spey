"""Logging style for spey"""
import logging
import sys
from contextlib import contextmanager

# pylint: disable=C0103


class ColoredFormatter(logging.Formatter):
    """Coloured logging formatter for spey"""

    def __init__(self, msg):
        logging.Formatter.__init__(self, msg)

    def format(self, record):
        if record.levelno >= 50:  # FATAL
            color = "\x1b[31mSpey - ERROR: "
        elif record.levelno >= 40:  # ERROR
            color = "\x1b[31mSpey - ERROR: "
        elif record.levelno >= 30:  # WARNING
            color = "\x1b[35mSpey - WARNING: "
        elif record.levelno >= 20:  # INFO
            color = "\x1b[0mSpey: "
        elif record.levelno >= 10:  # DEBUG
            color = (
                f"\x1b[36mSpey - DEBUG ({record.module}.{record.funcName}() "
                f"in {record.filename}::L{record.lineno}): "
            )
        else:  # ANYTHING ELSE
            color = "\x1b[0mSpey: "

        record.msg = color + str(record.msg) + "\x1b[0m"
        return logging.Formatter.format(self, record)


def init(LoggerStream=sys.stdout):
    """Initialise logger"""
    rootLogger = logging.getLogger()
    hdlr = logging.StreamHandler()
    fmt = ColoredFormatter("%(message)s")
    hdlr.setFormatter(fmt)
    rootLogger.addHandler(hdlr)

    # we need to replace all root loggers by ma5 loggers for a proper
    # interface with madgraph5
    SpeyLogger = logging.getLogger("Spey")
    for hdlr in SpeyLogger.handlers:
        SpeyLogger.removeHandler(hdlr)
    hdlr = logging.StreamHandler(LoggerStream)
    fmt = ColoredFormatter("%(message)s")
    hdlr.setFormatter(fmt)
    SpeyLogger.addHandler(hdlr)
    SpeyLogger.propagate = False


@contextmanager
def disable_logging(highest_level: int = logging.CRITICAL):
    """
    Temporary disable logging implementation

    Args:
        highest_level (``int``, default ``logging.CRITICAL``): highest level to be set in logging
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)


@contextmanager
def capture_logs(level: int = logging.DEBUG, stream=None):
    """
    Context manager that captures log records emitted while inside the context
    and suppresses duplicate messages. Only unique messages are printed once
    to the provided stream (defaults to sys.stdout).

    **Example:**

    .. code:: python3

        >>> with capture_logs(logging.INFO) as _:
        >>>     for i in range(100):
        >>>         logging.getLogger("Spey").warning("Same warning")  # printed once
        >>> # outputs: "Same warning" only once

    Args:
        level (``int``, default ``logging.DEBUG``): minimum logging level to capture.
        stream (default ``None``): file-like object to write the unique messages.
    """
    stream = stream or sys.stdout
    records = []
    seen_messages = set()

    class _DedupHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
            except Exception:
                msg = record.getMessage()

            # Only add if we haven't seen this exact message before
            if msg not in seen_messages:
                seen_messages.add(msg)
                records.append(msg)

    handler = _DedupHandler()
    handler.setFormatter(ColoredFormatter("%(message)s"))
    handler.setLevel(level)

    # Get the Spey logger
    spey_logger = logging.getLogger("Spey")
    prev_level = spey_logger.level
    prev_propagate = spey_logger.propagate

    # Store and remove existing handlers temporarily
    prev_handlers = spey_logger.handlers[:]
    for h in prev_handlers:
        spey_logger.removeHandler(h)

    spey_logger.addHandler(handler)
    spey_logger.setLevel(level)
    spey_logger.propagate = False

    try:
        yield
        # Write unique messages to the provided stream
        for m in records:
            stream.write(m + "\n")
        try:
            stream.flush()
        except Exception:
            pass
    finally:
        spey_logger.removeHandler(handler)
        spey_logger.setLevel(prev_level)
        spey_logger.propagate = prev_propagate
        # Restore original handlers
        for h in prev_handlers:
            spey_logger.addHandler(h)
