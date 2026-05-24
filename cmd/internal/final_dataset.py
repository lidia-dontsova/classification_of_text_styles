# Объединение всех стилей в один датасет и разбивка на train / val / test с стратификацией по классам

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Путь к файлам
BASE = Path('/Users/lidia.donts/Documents/учеба/python_projects/text-styles/cmd')

FILES = [
    BASE / 'datasets' / 'preprocessed' / 'scientific_preprocessed.csv',
    BASE / 'datasets' / 'preprocessed' / 'formal_preprocessed.csv',
    BASE / 'datasets' / 'preprocessed' / 'journalistic_preprocessed.csv',
    BASE / 'datasets' / 'preprocessed' / 'artistic_preprocessed.csv',
    BASE / 'datasets' / 'preprocessed' / 'conversational_preprocessed.csv',
]

OUT_DIR = BASE / 'datasets'

# Загрузка и объединение
dfs = []
for path in FILES:
    df = pd.read_csv(path)
    print(f'{path.name}: {len(df)} строк, label={df["label"].iloc[0]}, style={df["style"].iloc[0]}')
    dfs.append(df)

corpus = pd.concat(dfs, ignore_index=True)
print(f'\nВсего строк: {len(corpus)}')
print(f'Распределение по классам:')
print(corpus.groupby(['label', 'style_ru']).size().to_string())

# Стратифицированная разбивка
X = corpus['text_ready']
y = corpus['label']

# Сначала отделяем test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

# Из оставшихся 85% отделяем val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.15 / 0.85,
    stratify=y_temp,
    random_state=42
)

print(f'\nРазбивка:')
print(f'Train: {len(X_train)} ({len(X_train)/len(corpus)*100:.1f}%)')
print(f'Val:   {len(X_val)}  ({len(X_val)/len(corpus)*100:.1f}%)')
print(f'Test:  {len(X_test)}  ({len(X_test)/len(corpus)*100:.1f}%)')

# Проверяем стратификацию — доли классов должны быть одинаковы
print(f'\nДоли классов в train:')
print((y_train.value_counts(normalize=True).sort_index() * 100).round(1).to_string())
print(f'\nДоли классов в val:')
print((y_val.value_counts(normalize=True).sort_index() * 100).round(1).to_string())
print(f'\nДоли классов в test:')
print((y_test.value_counts(normalize=True).sort_index() * 100).round(1).to_string())

# Сборка датафреймов
def make_df(X, y):
    df = pd.DataFrame({'text_ready': X, 'label': y})
    df = df.merge(
        corpus[['text_ready', 'style', 'style_ru']].drop_duplicates('text_ready'),
        on='text_ready', how='left'
    )
    return df.reset_index(drop=True)

train_df = make_df(X_train, y_train)
val_df   = make_df(X_val,   y_val)
test_df  = make_df(X_test,  y_test)

# Сохранение
corpus.to_csv(OUT_DIR / 'corpus.csv',    index=False, encoding='utf-8-sig')
train_df.to_csv(OUT_DIR / 'train.csv',   index=False, encoding='utf-8-sig')
val_df.to_csv(OUT_DIR / 'val.csv',       index=False, encoding='utf-8-sig')
test_df.to_csv(OUT_DIR / 'test.csv',     index=False, encoding='utf-8-sig')

print(f'\nСохранено:')
print(f'corpus.csv  — {len(corpus)} строк (полный датасет)')
print(f'train.csv   — {len(train_df)} строк')
print(f'val.csv     — {len(val_df)} строк')
print(f'test.csv    — {len(test_df)} строк')