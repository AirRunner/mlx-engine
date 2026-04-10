"""Shared helpers for boundary detection and chat-template token resolution."""

from __future__ import annotations

from typing import Callable, List, Optional


# Known turn delimiter tokens by chat template family.
# Add new entries to support additional templates.
TURN_START_MARKERS = [
    "<|im_start|>",  # ChatML (Qwen, Mistral, etc.)
    "<|start_header_id|>",  # Llama 3+
]
TURN_END_MARKERS = [
    "<|im_end|>",  # ChatML
    "<|eot_id|>",  # Llama 3+
]


def resolve_marker_id(vocab: dict, markers: list) -> Optional[int]:
    """Return the token ID of the first marker found in *vocab*, or None."""
    for marker in markers:
        mid = vocab.get(marker)
        if mid is not None:
            return mid
    return None


def find_system_message_end(prompt_tokens: list, tokenizer) -> Optional[int]:
    """Return the index of the first turn-end token (end of system message)."""
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return None
    marker_id = resolve_marker_id(vocab, TURN_END_MARKERS)
    if marker_id is None:
        return None
    try:
        return prompt_tokens.index(marker_id)
    except ValueError:
        return None


def find_conversation_start(prompt_tokens: list, tokenizer) -> Optional[int]:
    """Return the index of the second turn-start token (first conversation turn)."""
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return None
    marker_id = resolve_marker_id(vocab, TURN_START_MARKERS)
    if marker_id is None:
        return None
    positions = [i for i, t in enumerate(prompt_tokens) if t == marker_id]
    return positions[1] if len(positions) >= 2 else None


def make_marker_detector(markers: List[str]) -> Callable:
    """Return a ``find_stable_boundary(prompt_tokens, tokenizer)`` function
    that cuts the system prompt just before the earliest ``markers`` match."""

    def find_stable_boundary(prompt_tokens: list, tokenizer) -> Optional[int]:
        sys_end = find_system_message_end(prompt_tokens, tokenizer)
        if sys_end is None or sys_end == 0:
            return None

        sys_tokens = prompt_tokens[:sys_end]

        try:
            sys_text = tokenizer.decode(sys_tokens)
        except Exception:
            return None

        earliest_pos: Optional[int] = None
        for marker in markers:
            pos = sys_text.find(marker)
            if pos >= 0 and (earliest_pos is None or pos < earliest_pos):
                earliest_pos = pos

        if earliest_pos is None:
            return None

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

    return find_stable_boundary
