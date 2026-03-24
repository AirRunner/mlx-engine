"""OpenCode boundary detector.

OpenCode appends a dynamic environment block at the end of the system prompt:

    <|im_start|>system
    # Tools
    ...tools + instructions...            <- STABLE (used as cache key)
    Here is some useful information about the environment you are running in:
    <env>
      Working directory: /path/to/dir     <- dynamic: changes per project
      Is directory a git repo: yes/no     <- dynamic
      Platform: darwin                    <- stable
      Today's date: Tue Mar 24 2026       <- dynamic: changes daily
    </env>
    ...
    <|im_end|>

The boundary is placed right before the
"\\nHere is some useful information about the environment" line.

Returns None when the marker is absent (non-OpenCode prompt).
"""

from mlx_engine.cache_plugins._base import make_marker_detector

find_stable_boundary = make_marker_detector(
    ["\nHere is some useful information about the environment you are running in:"]
)
