"""Open WebUI boundary detector.

Open WebUI appends dynamic content to the system prompt on every turn:

    <|im_start|>system
    # Tools
    ...tools + format instructions...   <- STABLE (used as cache key)

    System time context: <date/time>    <- dynamic: changes daily or per-second
    ---
    The following is a list of stored memories...  <- dynamic: grows per turn
    [{"content": ...}, ...]
    <|im_end|>

The boundary is the earliest of:
  1. "\\nSystem time context:" — covers date/time and everything after
  2. "\\n---\\n"              — fallback when the time line is absent

Returns None when neither marker is found (non-Open-WebUI prompt), so the
generic ChatML fallback in cache_wrapper takes over with no regression.
"""

from mlx_engine.cache_plugins._base import make_marker_detector

find_stable_boundary = make_marker_detector(["\nSystem time context:", "\n---\n"])
