# Разбиение текстов на фрагменты

import re

MIN_LEN = 800
TARGET_LEN = 1500
MAX_LEN = 3000

SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
PARA_SPLIT = re.compile(r"\n\s*\n+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]")
_LETTER = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]")
# перенос слова: буква + управляющий символ + перевод строки + продолжение
_SOFT_BREAK = re.compile(
    r"([a-zA-Zа-яА-ЯёЁ0-9])[\x00-\x1f\x7f-\x9f]*\n[\x00-\x1f\x7f-\x9f]*"
    r"([a-zа-яё0-9]{1,5})(?![a-zа-яё0-9])"
)
# cp1251, ошибочно прочитанный как Latin-1
_MOJIBAKE_RUN = re.compile(
    r"[\u00c0-\u00ff](?:[\u00c0-\u00ff\-]*[\u00c0-\u00ff])"
    r"(?:\s+[\u00c0-\u00ff](?:[\u00c0-\u00ff\-]*[\u00c0-\u00ff])?)*"
    r"|[\u00c0-\u00ff]{5,}"
)


def _fix_mojibake_run(run: str) -> str:
    try:
        fixed = run.encode("latin-1").decode("cp1251")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return run
    cyr = sum(1 for c in fixed if "\u0400" <= c <= "\u04ff")
    n = len(fixed)
    if n == 0:
        return run
    latin_high = sum(1 for c in run if "\u00c0" <= c <= "\u00ff")
    if n <= 4 and latin_high == n and cyr == n:
        return fixed
    if cyr >= max(3, n * 0.4):
        return fixed
    return run


def fix_mojibake(text: str) -> str:
    return _MOJIBAKE_RUN.sub(lambda m: _fix_mojibake_run(m.group(0)), text)


def clean_text(text: str) -> str:
    # Убрать артефакты PDF и битую кодировку, сохранив слова.
    text = fix_mojibake(text)
    text = _SOFT_BREAK.sub(r"\1\2", text)

    def replace_control(match: re.Match[str]) -> str:
        i = match.start()
        before = text[i - 1] if i else ""
        after = text[i + 1] if i + 1 < len(text) else ""
        if _LETTER.match(before) and _LETTER.match(after):
            return "-"
        return ""

    text = _CONTROL.sub(replace_control, text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_long_block(block: str) -> list[str]:
    parts = SENT_SPLIT.split(block.strip())
    chunks: list[str] = []
    current = ""

    for sent in parts:
        sent = sent.strip()
        if not sent:
            continue

        candidate = f"{current} {sent}".strip() if current else sent
        if len(candidate) <= MAX_LEN:
            current = candidate
            continue

        if current and len(current) >= MIN_LEN:
            chunks.append(current)

        if len(sent) > MAX_LEN:
            for i in range(0, len(sent), MAX_LEN):
                piece = sent[i : i + MAX_LEN].strip()
                if len(piece) >= MIN_LEN:
                    chunks.append(piece)
            current = ""
        else:
            current = sent

    if current and len(current) >= MIN_LEN:
        chunks.append(current)

    return chunks


def chunk_text(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    if len(text) <= MAX_LEN:
        return [text] if len(text) >= MIN_LEN else []

    paragraphs = [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > MAX_LEN:
            if current and len(current) >= MIN_LEN:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_block(para))
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= TARGET_LEN:
            current = candidate
            continue

        if current and len(current) >= MIN_LEN:
            chunks.append(current)

        if len(para) <= MAX_LEN:
            current = para if len(para) >= MIN_LEN else ""
        else:
            chunks.extend(split_long_block(para))
            current = ""

    if current and len(current) >= MIN_LEN:
        chunks.append(current)

    return chunks
