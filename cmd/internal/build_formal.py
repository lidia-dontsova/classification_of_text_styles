# Сборка датасета официально-деловых текстов

import csv
import random
import re
from pathlib import Path

TXT_PATH     = Path('/Users/lidia.donts/Documents/учеба/python_projects/text-styles/cmd/draft/formal-buisnes/new_train_dataset.txt')
OUT_PATH     = Path('/Users/lidia.donts/Documents/учеба/python_projects/text-styles/cmd/datasets/formal_2000.csv')

MIN_LEN      = 400
TARGET_LEN   = 1000
MAX_LEN      = 3000
SAMPLE_SIZE  = 2000
RANDOM_STATE = 42

PARA_SPLIT = re.compile(r'\n{2,}')

def read_txt(path: Path) -> str:
    for enc in ('utf-8', 'cp1251', 'utf-8-sig'):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return ''

# Разбиваем по началу новой статьи
DOC_SPLIT = re.compile(r'(?=Статья\s+\d+[\.\s]|ГЛАВА\s+\d+|Раздел\s+[IVX]+|РОССИЙСКАЯ ФЕДЕРАЦИЯ)')

def chunk_text(text: str) -> list[str]:
    text = re.sub(r'[ \t]+', ' ', text).strip()
    if not text:
        return []

    # Разбиваем по маркерам начала статей
    parts = [p.strip() for p in DOC_SPLIT.split(text) if p.strip()]
    print(f'  Частей после split: {len(parts)}')

    chunks, current = [], ''
    for part in parts:
        candidate = f'{current} {part}'.strip() if current else part
        if len(candidate) <= TARGET_LEN:
            current = candidate
        else:
            if current and len(current) >= MIN_LEN:
                chunks.append(current)
            current = part if len(part) >= MIN_LEN else ''
    if current and len(current) >= MIN_LEN:
        chunks.append(current)
    return chunks

def main():
    text = read_txt(TXT_PATH)
    if not text:
        print('Файл не читается')
        return

    print(f'Размер файла: {len(text):,} ')

    all_chunks = chunk_text(text)
    print(f'Всего фрагментов: {len(all_chunks)}')

    if not all_chunks:
        print('Фрагментов нет ')
        return

    if len(all_chunks) < SAMPLE_SIZE:
        sampled = all_chunks
        print(f'Фрагментов меньше {SAMPLE_SIZE} — берём все: {len(sampled)}')
    else:
        rng = random.Random(RANDOM_STATE)
        sampled = rng.sample(all_chunks, SAMPLE_SIZE)
        print(f'Случайная выборка: {len(sampled)}')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label', 'style', 'style_ru'])
        writer.writeheader()
        for t in sampled:
            writer.writerow({
                'text':     t,
                'label':    1,
                'style':    'official',
                'style_ru': 'Официально-деловой',
            })
            
if __name__ == '__main__':
    main()