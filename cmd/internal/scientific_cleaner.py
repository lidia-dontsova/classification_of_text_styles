# Предобработка научных текстов

import pandas as pd
import re
import pymorphy3 as pymorphy2

# список стоп-слов
STOPWORDS = {
    'и','в','не','на','что','я','с','он','как','это','по','но','от','к',
    'а','за','же','так','его','из','до','мне','они','у','была','был','было',
    'бы','есть','быть','при','уже','или','нет','нас','вот','для','вы','мы',
    'то','все','она','ее','её','ли','об','если','ещё','их','вас','вам',
    'им','тем','между','когда','там','потому','ним','нем','под','над','чтобы',
    'со','без','во','ты','тебя','тебе','ему','нам','них','ними',
    'этот','эта','эти','тот','та','те','свой','своя','свое','свои',
    'весь','вся','всё','всех','который','которая','которое','которые',
    'кто','где','куда','откуда','почему','зачем','поэтому','однако',
    'также','тоже','даже','лишь','только','уж','ну','ведь','же','разве',
    # Английские стоп-слова (для библиографических ссылок)
    'the','and','for','are','was','with','this','that','from',
    'has','have','been','not','but','they','its','can','all',
}

# Удаление артефактов датасета
def remove_artifacts(text: str) -> str:
    text = re.sub(r'\w*token\w*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b33\b', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\[\d+\]', ' ', text)    # ссылки типа [1], [23]
    text = re.sub(r'\(\d{4}\)', ' ', text)  # годы типа (2002)
    return text

# Нормализация 
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^а-яёa-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Токенизация
def tokenize(text: str) -> list:
    return text.split()

# Удаление стоп-слов
def remove_stopwords(tokens: list, min_len: int = 3) -> list:
    return [t for t in tokens if t not in STOPWORDS and len(t) >= min_len]

# Лемматизация 
_morph = pymorphy2.MorphAnalyzer()

def lemmatize(word: str) -> str:
    return _morph.parse(word)[0].normal_form

def preprocess(text: str) -> str:
    text = remove_artifacts(text)
    text = normalize_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = [lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Запуск на датасете
if __name__ == '__main__':
    df = pd.read_csv('/Users/lidia.donts/Documents/учеба/python_projects/text-styles/cmd/datasets/scientific_2000.csv')

    df['text_ready']  = df['text'].astype(str).apply(preprocess)
    df['token_count'] = df['text_ready'].str.split().str.len()

    before = len(df)
    df = df[df['token_count'] >= 5].reset_index(drop=True)

    print(f'Строк: {len(df)} (удалено коротких: {before - len(df)})')
    print(f'Средн. токенов: {df["token_count"].mean():.1f}')

    df[['text_ready', 'label', 'style', 'style_ru']].to_csv(
        'scientific_preprocessed.csv',
        index=False, encoding='utf-8-sig'
    )
    print('Сохранено: scientific_preprocessed.csv')
