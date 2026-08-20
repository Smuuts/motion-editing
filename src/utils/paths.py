"""
Repo-relative path resolution.

Checkpoint configs record paths as they were typed at training time, i.e. relative to
the repo root — so resolving them against the current working directory silently picks
up the wrong file whenever a script is run from elsewhere.
"""

import os

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)


def repo_path(*parts: str) -> str:
    """Absolute path to `parts` under the repo root, independent of the cwd."""
    return os.path.join(REPO_ROOT, *parts)


def resolve_repo_path(path: str) -> str:
    """An absolute path unchanged; a relative one resolved against the repo root."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
