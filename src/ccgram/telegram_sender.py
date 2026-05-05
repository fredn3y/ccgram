"""Message splitting utility for Telegram's 4096-character limit.

Provides:
  - split_message(): splits long text into Telegram-safe chunks (≤4096 chars),
    preferring newline boundaries.
"""

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
TELEGRAM_SAFE_RENDERED_LENGTH = 3900


def split_message(
    text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH
) -> list[str]:
    """Split a message into chunks that fit Telegram's length limit.

    Tries to split on newlines when possible to preserve formatting.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = ""

    for line in text.split("\n"):
        # If single line exceeds max, split it forcefully
        if len(line) > max_length:
            if current_chunk:
                chunks.append(current_chunk.rstrip("\n"))
                current_chunk = ""
            # Split long line into fixed-size pieces
            for i in range(0, len(line), max_length):
                chunks.append(line[i : i + max_length])
        elif len(current_chunk) + len(line) + 1 > max_length:
            # Current chunk is full, start a new one
            chunks.append(current_chunk.rstrip("\n"))
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"

    if current_chunk:
        chunks.append(current_chunk.rstrip("\n"))

    return chunks


def split_rendered_message(
    text: str,
    max_length: int = TELEGRAM_SAFE_RENDERED_LENGTH,
) -> list[str]:
    """Split text by Telegram's final rendered length after entity conversion."""
    chunks: list[str] = []
    for chunk in split_message(text, max_length=max_length):
        chunks.extend(_split_chunk_by_rendered_size(chunk, max_length))
    return chunks


def _split_chunk_by_rendered_size(text: str, max_length: int) -> list[str]:
    if _rendered_length(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        split_at = _largest_rendered_prefix(remaining, max_length)
        if split_at <= 0:
            split_at = min(len(remaining), max_length)

        newline_at = remaining.rfind("\n", 0, split_at + 1)
        if newline_at > 0 and newline_at >= split_at // 2:
            split_at = newline_at + 1

        chunk = remaining[:split_at].rstrip("\n")
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip("\n")

    return chunks or [text[:max_length]]


def _largest_rendered_prefix(text: str, max_length: int) -> int:
    low = 0
    high = len(text)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            low = 1
            continue
        if _rendered_length(text[:mid]) <= max_length:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def _rendered_length(text: str) -> int:
    from telegramify_markdown import utf16_len

    from .entity_formatting import convert_to_entities

    plain_text, _entities = convert_to_entities(text)
    return utf16_len(plain_text)
