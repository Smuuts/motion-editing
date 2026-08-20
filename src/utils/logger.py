"""
One logger for the whole project: console, run log file, JSONL metrics and W&B.

    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Device: %s", device)
    log.section("Stage 2 — masking")
    log.table(["scale", "R@1"], rows)

Entry points call `configure(...)` once (usually via `utils.cli`) to set the verbosity
and, optionally, a file copy of everything printed. Long-running jobs additionally
attach a metrics sink with `log.attach_run(output_dir, use_wandb=...)`, after which
`log.metrics({...})` appends a JSON line per record and mirrors it to W&B.

Console writes go through `tqdm.write` when a progress bar is on screen, so messages
never land in the middle of a bar.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Iterable, Mapping

from utils.log_format import key_values, rule, section, table

ROOT_NAME = "motion"
LEVELS = {"debug": logging.DEBUG, "info": logging.INFO,
          "warning": logging.WARNING, "error": logging.ERROR}

_configured = False
_console_level = logging.INFO


class _TqdmHandler(logging.StreamHandler):
    """Stream handler that defers to tqdm.write so bars are not overwritten."""

    def emit(self, record):
        try:
            from tqdm import tqdm
            tqdm.write(self.format(record), file=self.stream)
        except ImportError:
            super().emit(record)
        except Exception:                              # pragma: no cover - logging path
            self.handleError(record)


def configure(level: str = "info", log_file: str | None = None,
              timestamps: bool = False) -> None:
    """Set up the process-wide console (and optional file) sink. Idempotent.

    Messages are printed bare by default — this project's output is read as a report,
    not as a service log — with `timestamps=True` prefixing "HH:MM:SS LEVEL" instead.

    `level` throttles the CONSOLE only. A `log_file` always records everything down to
    DEBUG, so turning the console down with --quiet never costs you the run's record,
    which is the whole reason to ask for a file in the first place.
    """
    global _configured, _console_level
    _console_level = LEVELS.get(level, logging.INFO)

    root = logging.getLogger(ROOT_NAME)
    root.setLevel(logging.DEBUG if log_file else _console_level)
    root.propagate = False
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)

    fmt = "%(asctime)s %(levelname)-7s %(message)s" if timestamps else "%(message)s"
    console = _TqdmHandler(stream=sys.stderr)
    console.setLevel(_console_level)
    console.setFormatter(_LevelFormatter(fmt, datefmt="%H:%M:%S"))
    root.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s  %(message)s"))
        root.addHandler(file_handler)
    _configured = True


class _LevelFormatter(logging.Formatter):
    """Tags warnings and errors inline; info/debug stay unadorned."""

    _TAGS = {logging.WARNING: "WARNING: ", logging.ERROR: "ERROR: ",
             logging.CRITICAL: "ERROR: "}

    def format(self, record):
        message = super().format(record)
        tag = self._TAGS.get(record.levelno, "")
        return f"{tag}{message}" if tag else message


class Logger:
    """Console helpers plus an optional per-run metrics sink.

    The console methods (`info`/`warning`/`section`/`kv`/`table`/`wrote`) are always
    available. `metrics()` is a no-op until `attach_run()` names an output directory.
    """

    def __init__(self, name: str = ROOT_NAME):
        if not _configured:
            configure()
        self._log = logging.getLogger(f"{ROOT_NAME}.{name}" if name != ROOT_NAME else name)
        self._metrics_path: str | None = None
        self._wandb = None

    # ── run sink ────────────────────────────────────────────────────────────────
    def attach_run(self, output_dir: str, use_wandb: bool = False,
                   project: str = "motion-dit", run_name: str | None = None,
                   filename: str = "metrics.jsonl") -> "Logger":
        """Point `metrics()` at `<output_dir>/<filename>`, optionally mirroring to W&B."""
        os.makedirs(output_dir, exist_ok=True)
        self._metrics_path = os.path.join(output_dir, filename)
        if use_wandb:
            try:
                import wandb
                wandb.init(project=project, name=run_name or output_dir)
                self._wandb = wandb
            except ImportError:
                self.warning("wandb not installed — logging metrics to file only.")
        return self

    def metrics(self, record: Mapping[str, Any]) -> None:
        """Append one timestamped JSON line, and mirror it to W&B when enabled."""
        if self._metrics_path is None:
            return
        payload = {**record, "timestamp": datetime.now().isoformat()}
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(payload) + "\n")
        if self._wandb is not None:
            self._wandb.log(dict(record))

    # ── console ─────────────────────────────────────────────────────────────────
    def debug(self, msg: str, *args) -> None:
        self._log.debug(msg, *args)

    def info(self, msg: str = "", *args) -> None:
        self._log.info(msg, *args)

    def warning(self, msg: str, *args) -> None:
        self._log.warning(msg, *args)

    def error(self, msg: str, *args) -> None:
        self._log.error(msg, *args)

    def blank(self) -> None:
        self._log.info("")

    def section(self, title: str) -> None:
        """A blank line and a titled rule — the separator between phases of a run."""
        self._log.info(section(title))

    def rule(self, title: str | None = None) -> None:
        self._log.info(rule(title))

    def kv(self, mapping: Mapping[str, Any], indent: int = 2,
           precision: int = 4) -> None:
        """Aligned `key : value` block."""
        rendered = key_values(mapping, indent=indent, precision=precision)
        if rendered:
            self._log.info(rendered)

    def table(self, headers, rows, title: str | None = None, indent: int = 2,
              precision: int = 4, align: str | None = None) -> None:
        if title:
            self._log.info(title)
        self._log.info(table(headers, rows, indent=indent, precision=precision,
                             align=align))

    def wrote(self, path: str, what: str = "") -> None:
        """Standard "output landed here" line, so every script announces files alike."""
        self._log.info("wrote %s%s", path, f"  ({what})" if what else "")

    def progress(self, iterable: Iterable, desc: str = "", **kwargs):
        """tqdm over `iterable`, silenced when the CONSOLE is quieter than INFO.

        Keyed to the console level, not the logger's: a run writing a --log_file lowers
        the logger to DEBUG, and progress bars must not come back on because of it.
        """
        from tqdm import tqdm
        kwargs.setdefault("leave", False)
        kwargs.setdefault("disable", _console_level > logging.INFO)
        return tqdm(iterable, desc=desc, **kwargs)


_loggers: dict[str, Logger] = {}


def get_logger(name: str = ROOT_NAME) -> Logger:
    """The Logger for `name` (pass `__name__`); one instance per name per process."""
    short = name.rsplit(".", 1)[-1] if name != ROOT_NAME else ROOT_NAME
    if short not in _loggers:
        _loggers[short] = Logger(short)
    return _loggers[short]


# ── CLI wiring ──────────────────────────────────────────────────────────────────
# Defined here rather than in utils/cli.py so the stdlib-only scripts can take the same
# flags without pulling in torch. utils.cli re-exports them.

def add_logging_args(parser):
    """--log_level / --log_file / --quiet, for `configure_logging` to apply."""
    parser.add_argument("--log_level", default="info", choices=sorted(LEVELS),
                        help="Console verbosity (default info).")
    parser.add_argument("--log_file", default=None,
                        help="Also write every message to this file, with timestamps "
                             "and module names.")
    parser.add_argument("--quiet", action="store_true",
                        help="Warnings and errors only; also hides progress bars. "
                             "Shorthand for --log_level warning.")
    return parser


def configure_logging(args):
    """Apply the logging flags and return `args`, so it can wrap `parse_args()`.

    Safe on a parser that never defined them — the defaults then apply.
    """
    level = ("warning" if getattr(args, "quiet", False)
             else getattr(args, "log_level", "info"))
    configure(level=level, log_file=getattr(args, "log_file", None))
    return args
