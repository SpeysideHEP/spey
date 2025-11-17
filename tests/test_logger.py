import io
import logging

from spey.system import logger as slogger


def test_colored_formatter_formats_levels():
    fmt = slogger.ColoredFormatter("%(message)s")

    rec_err = logging.LogRecord(
        "Spey", logging.ERROR, "somefile.py", 123, "error-message", (), None
    )
    rec_err.module = "mod"
    rec_err.funcName = "fn"
    rec_err.filename = "somefile.py"
    rec_err.lineno = 123

    out_err = fmt.format(rec_err)
    # should include the coloured prefix and the human-readable "Spey - ERROR" text
    assert "\x1b[" in out_err
    assert "Spey - ERROR" in out_err

    rec_info = logging.LogRecord(
        "Spey", logging.INFO, "somefile.py", 10, "info-message", (), None
    )
    rec_info.module = "mod2"
    rec_info.funcName = "fn2"
    rec_info.filename = "somefile.py"
    rec_info.lineno = 10

    out_info = fmt.format(rec_info)
    assert "\x1b[" in out_info
    assert "Spey:" in out_info  # INFO uses "Spey:" prefix


def test_init_attaches_handlers_and_writes_to_stream():
    root = logging.getLogger()
    spey_logger = logging.getLogger("Spey")

    # preserve original state so test does not leak global changes
    orig_root_handlers = list(root.handlers)
    orig_spey_handlers = list(spey_logger.handlers)
    orig_propagate = spey_logger.propagate

    stream = io.StringIO()
    try:
        # initialize logger to write into our stream
        slogger.init(LoggerStream=stream)

        spey_logger = logging.getLogger("Spey")
        spey_logger.setLevel(logging.INFO)
        spey_logger.info("test-output")

        contents = stream.getvalue()
        assert "test-output" in contents
        # output should include colour escape sequences injected by ColoredFormatter
        assert "\x1b[" in contents
        # propagate should be disabled for the "Spey" logger
        assert spey_logger.propagate is False
    finally:
        # restore original handlers and propagate flag to avoid polluting other tests
        root.handlers = orig_root_handlers
        spey_logger.handlers = orig_spey_handlers
        spey_logger.propagate = orig_propagate
