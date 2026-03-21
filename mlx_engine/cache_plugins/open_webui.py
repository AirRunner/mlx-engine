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

from __future__ import annotations

from typing import Optional

_MARKERS = ["\nSystem time context:", "\n---\n"]


def find_stable_boundary(prompt_tokens: list, tokenizer) -> Optional[int]:
    """Return the token index where Open WebUI's dynamic suffix begins.

    The disk cache key becomes ``prompt_tokens[:result]``, containing only the
    stable tools + instructions block — identical across all turns and sessions.

    Uses binary search over ``tokenizer.decode`` to map the character-level
    marker position back to a token index (handles any BPE tokenizer cleanly).
    """
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return None

    im_end_id = vocab.get("<|im_end|>")
    if im_end_id is None:
        return None

    # Locate the end of the system prompt block.
    try:
        sys_end = prompt_tokens.index(im_end_id)
    except ValueError:
        return None

    if sys_end == 0:
        return None

    sys_tokens = prompt_tokens[:sys_end]

    # Decode the full system prompt once.
    try:
        sys_text = tokenizer.decode(sys_tokens)
    except Exception:
        return None

    # Find the earliest dynamic marker.
    earliest_pos: Optional[int] = None
    for marker in _MARKERS:
        pos = sys_text.find(marker)
        if pos >= 0 and (earliest_pos is None or pos < earliest_pos):
            earliest_pos = pos

    if earliest_pos is None:
        return None  # Not an Open WebUI prompt — let the generic fallback handle it.

    # Binary-search for the token index whose decoded prefix first exceeds
    # earliest_pos characters.  Handles split/multi-byte tokens correctly.
    lo, hi = 0, len(sys_tokens)
    while lo < hi:
        mid = (lo + hi) // 2
        try:
            decoded_len = len(tokenizer.decode(sys_tokens[:mid]))
        except Exception:
            return None
        if decoded_len <= earliest_pos:
            lo = mid + 1
        else:
            hi = mid

    return lo
