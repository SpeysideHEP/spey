"""Logging style for spey"""
import logging
import sys

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
