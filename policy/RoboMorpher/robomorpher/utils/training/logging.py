"""Rank-zero plain file logs and starVLA-style terminal progress.

Redirect stdout to a log file to retain plain records; stderr (or the controlling
TTY when a launcher pipes both streams) receives coloured human-readable output.
Do not additionally tee stdout to the same terminal or INFO records appear twice.
EAT_CONSOLE_TTY=$(tty) preserves the launcher terminal through torchrun sessions.
EAT_CONSOLE=plain disables all terminal mirroring and dynamic progress.
"""
from datetime import datetime
import logging
import random
import numbers
import os
import sys
from tqdm import tqdm
from rich.console import Console
from rich.logging import RichHandler

_active_console = None


def _value(value):
    if isinstance(value, numbers.Real) and not isinstance(value, (int, bool)):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_value(v) for v in value) + "]"
    return str(value)


def _is_tty(stream):
    try:
        return stream.isatty()
    except (AttributeError, OSError):
        return False


def _open_tty_path(path):
    # Do not create or truncate a mistakenly configured regular file. O_NOCTTY
    # also avoids assigning a controlling terminal to torchrun session leaders.
    descriptor = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_NOCTTY | os.O_NONBLOCK)
        if not os.isatty(descriptor):
            os.close(descriptor)
            return None
        os.set_blocking(descriptor, True)
        return os.fdopen(descriptor, "w", buffering=1)
    except (OSError, ValueError):
        if descriptor is not None:
            os.close(descriptor)
        return None


def _open_terminal():
    if os.environ.get("EAT_CONSOLE", "auto").lower() in {"plain", "off", "0"}:
        return None, False
    if _is_tty(sys.stderr):
        return sys.stderr, False
    if _is_tty(sys.stdout):
        return sys.stdout, False
    # torchrun starts workers in new sessions: inherited pipes are not TTYs,
    # and /dev/tty cannot open without a controlling terminal. The launcher may
    # explicitly pass its known `tty` path; never guess another user's terminal.
    terminal_path = os.environ.get("EAT_CONSOLE_TTY")
    if terminal_path:
        stream = _open_tty_path(terminal_path)
        if stream is not None:
            return stream, True
    stream = _open_tty_path("/dev/tty")
    return (stream, True) if stream is not None else (None, False)


# Formatting matches starVLA/training/trainer_utils/overwatch.py at
# a376315d328b7c47a18497f09ef59ecbc37e37d1; handlers remain private instead of
# replacing the application's global logging configuration.
RICH_FORMATTER, DATEFMT = "| >> %(message)s", "%m/%d [%H:%M:%S]"


def _handler(stream):
    console = Console(file=stream, no_color=os.environ.get("NO_COLOR") is not None)
    handler = RichHandler(console=console, markup=True, rich_tracebacks=True,
                          show_level=True, show_path=True, show_time=True)
    handler.setFormatter(logging.Formatter(RICH_FORMATTER, datefmt=DATEFMT))
    return handler


def _caller_record(message):
    # Capture the actual caller: native trainer or a read-only replay script.
    frame = sys._getframe(2)
    return logging.LogRecord("eat.training", logging.INFO, frame.f_code.co_filename,
                             frame.f_lineno, message, (), None, func=frame.f_code.co_name)


def _render_record(handler, record, terminal):
    # Rich may lazily allocate style IDs using Python RNG. Console rendering
    # must not perturb augmentation or exact resume random streams.
    state = random.getstate()
    try:
        with tqdm.external_write_mode(file=terminal):
            handler.handle(record)
    finally:
        random.setstate(state)


def _emit(record, plain):
    if _active_console is not None:
        _active_console.write_record(record, plain)
        return
    terminal, owns_terminal = _open_terminal()
    if terminal is None:
        print(plain, flush=True)
        return
    if not _is_tty(sys.stdout):
        print(plain, flush=True)
    state = random.getstate()
    try:
        _render_record(_handler(terminal), record, terminal)
    finally:
        random.setstate(state)
        if owns_terminal:
            terminal.close()


class TrainingProgress:
    """The upstream plain tqdm layout, coordinated with private Rich logging."""
    def __init__(self, *, total, initial=0, rank=0):
        global _active_console
        self.terminal = None
        self.owns_terminal = False
        self.bar = None
        self.handler = None
        self.rank = rank
        if rank != 0:
            return
        state = random.getstate()
        try:
            self.terminal, self.owns_terminal = _open_terminal()
            if self.terminal is not None:
                self.handler = _handler(self.terminal)
                # Upstream tqdm defaults: no coloured bar or description.
                self.bar = tqdm(total=total, initial=initial, file=self.terminal)
            _active_console = self
        finally:
            random.setstate(state)

    def write_record(self, record, plain):
        if self.terminal is None:
            print(plain, flush=True)
            return
        if not _is_tty(sys.stdout):
            print(plain, flush=True)
        _render_record(self.handler, record, self.terminal)

    def update(self, *, data_times=None, model_times=None, advance=1, epoch=None, loss=None):
        # epoch/loss arguments remain accepted but are not postfix keys.
        if self.bar is not None:
            self.bar.update(advance)
            postfix = {}
            if data_times is not None:
                postfix["data_times"] = f"{data_times:.3f}"
            if model_times is not None:
                postfix["model_times"] = f"{model_times:.3f}"
            if postfix:
                self.bar.set_postfix(postfix)

    def close(self):
        global _active_console
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        if self.handler is not None:
            self.handler.close()
            self.handler = None
        if self.owns_terminal and self.terminal is not None:
            self.terminal.close()
        self.terminal = None
        self.owns_terminal = False
        if _active_console is self:
            _active_console = None


def log_info(message="", *, task="tokenizer", phase="setup", rank=0, **fields):
    """Preserve existing plain file records; render terminal INFO via Rich."""
    if rank != 0:
        return
    details = " | ".join(f"{key}={_value(value)}" for key, value in fields.items() if value is not None)
    body = " | ".join(part for part in (message, details) if part)
    message = f"{task}/{phase} | {body}"
    plain = f"{datetime.now():%Y-%m-%d %H:%M:%S} | INFO | {message}"
    _emit(_caller_record(message), plain)


def log_train_metrics(step, metrics, *, rank=0):
    """Render the real metric mapping; never rename, mutate or invent metrics."""
    if rank != 0:
        return
    message = f"Step {step}, Loss: {dict(metrics)}"
    plain = f"{datetime.now():%Y-%m-%d %H:%M:%S} | INFO | {message}"
    _emit(_caller_record(message), plain)
