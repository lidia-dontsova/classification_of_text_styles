# Сбор датасета художественных текстов

import csv
import random
from pathlib import Path

from dataset_chunking import chunk_text

ROOT = Path(__file__).resolve().parents[1]
ARTISTIC_DIR = ROOT / "draft" / "artistic"
OUT_PATH = ROOT / "datasets" / "artistic_2000.csv"

SAMPLE_SIZE = 2000
RANDOM_STATE = 42


def main() -> None:
    all_chunks: list[str] = []
    for path in sorted(ARTISTIC_DIR.rglob("*.txt")):
        all_chunks.extend(chunk_text(path.read_text(encoding="utf-8")))

    if len(all_chunks) < SAMPLE_SIZE:
        raise ValueError(
            f"Недостаточно фрагментов: {len(all_chunks)} < {SAMPLE_SIZE}"
        )

    rng = random.Random(RANDOM_STATE)
    sampled = rng.sample(all_chunks, SAMPLE_SIZE)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["text", "label", "style", "style_ru"]
        )
        writer.writeheader()
        for text in sampled:
            writer.writerow(
                {
                    "text": text,
                    "label": 2,
                    "style": "artistic",
                    "style_ru": "Художественный",
                }
            )

    print(f"Фрагментов всего: {len(all_chunks)}")

if __name__ == "__main__":
    main()
