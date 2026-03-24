"""Claude Code (rclaude) boundary detector.

The rclaude wrapper appends a dynamic environment section at the end of the
system prompt:

    <|im_start|>system
    ...tools + Claude Code instructions...   <- STABLE (used as cache key)
    # claude.md content                      <- stable

    # Environment                            <- boundary
     - Working directory: /path/to/dir       <- dynamic: changes per invocation
     - Is a git repository: true/false       <- dynamic
     - Platform: Darwin                      <- stable
     - OS Version: ...                       <- stable
     - Shell: /bin/zsh                       <- stable
    <|im_end|>

Note: the tool list may vary between sessions when LSP servers are active
(LSP tool inserted at a different position by Claude Code internally). This
results in a separate cache entry for the first request (no LSP) vs.
subsequent ones (LSP loaded) — unavoidable without changes to Claude Code.

Returns None when the marker is absent (non-rclaude prompt).
"""

from mlx_engine.cache_plugins._base import make_marker_detector

find_stable_boundary = make_marker_detector(["\n# Environment\n"])
