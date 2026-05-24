# Демонстрация классификатора стилей с выбором модели

import torch
import joblib
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

MODELS_DIR = Path('/Users/lidia.donts/Documents/учеба/python_projects/text-styles/cmd/models')

STYLES = {
    0: 'Научный',
    1: 'Официально-деловой',
    2: 'Публицистический',
    3: 'Разговорный',
    4: 'Художественный',
}

STOPWORDS = {
    'и','в','не','на','что','я','с','он','как','это','по','но','от','к',
    'а','за','же','так','его','из','до','мне','они','у','была','был','было',
    'бы','есть','быть','при','уже','или','нет','нас','вот','для','вы','мы',
    'то','все','она','ее','её','ли','об','если','ещё','их','вас','вам',
    'им','тем','между','когда','там','потому','ним','нем','под','над','чтобы',
    'со','без','во','ты','тебя','тебе','ему','нам','них','ними',
}

# предобработка (для классических моделей)
def preprocess(text: str) -> str:
    text = re.sub(r'https?://\S+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^а-яёa-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) >= 3]
    return ' '.join(tokens)

# загрузка моделей
def load_classical(model_name: str):
    tfidf = joblib.load(MODELS_DIR / 'tfidf_vectorizer.pkl')
    model = joblib.load(MODELS_DIR / f'{model_name}.pkl')
    return tfidf, model

def load_bert():
    path      = MODELS_DIR / 'rubert_best'
    tokenizer = AutoTokenizer.from_pretrained(path)
    model     = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

# предсказание
def predict_classical(text: str, tfidf, model) -> dict:
    text_clean = preprocess(text)
    vec        = tfidf.transform([text_clean])
    pred       = model.predict(vec)[0]

    # Вероятности
    if hasattr(model, 'predict_proba'):
        probs     = model.predict_proba(vec)[0]
        all_probs = {STYLES[i]: round(probs[i] * 100, 1) for i in range(5)}
    else:
        # LinearSVC
        scores    = model.decision_function(vec)[0]
        exp_s     = [2 ** s for s in scores]
        total     = sum(exp_s)
        all_probs = {STYLES[i]: round(exp_s[i] / total * 100, 1) for i in range(5)}

    return {
        'predicted':  STYLES[pred],
        'confidence': all_probs[STYLES[pred]],
        'all_probs':  all_probs,
    }

def predict_bert(text: str, tokenizer, model) -> dict:
    inputs = tokenizer(
        text,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)[0]

    pred_id   = probs.argmax().item()
    all_probs = {STYLES[i]: round(probs[i].item() * 100, 1) for i in range(5)}

    return {
        'predicted':  STYLES[pred_id],
        'confidence': round(probs[pred_id].item() * 100, 1),
        'all_probs':  all_probs,
    }

# вывод результата
def print_result(result: dict):
    print(f'\n{"─"*52}')
    print(f'  Стиль: {result["predicted"]}')
    print(f'  Уверенность: {result["confidence"]}%')
    print(f'\n  Вероятности по всем стилям:')
    for style, prob in sorted(result['all_probs'].items(),
                               key=lambda x: x[1], reverse=True):
        filled = int(prob / 5)
        bar    = '█' * filled + '░' * (20 - filled)
        print(f'  {style:<25} {bar} {prob:.1f}%')
    print(f'{"─"*52}\n')

# выбор модели
def choose_model():
    print('\n' + '='*52)
    print('  КЛАССИФИКАТОР СТИЛЕЙ ТЕКСТА')
    print('='*52)
    print('\nДоступные модели:\n')
    print('  1. Naive Bayes      (F1=0.9523)')
    print('  2. SVM              (F1=0.9719)')
    print('  3. Random Forest    (F1=0.9308)')
    print('  4. RuBERT           (F1=0.9899)')
    print()

    while True:
        choice = input('Выбери модель (1-4): ').strip()
        if choice == '1':
            return 'nb', load_classical('naive_bayes')
        elif choice == '2':
            return 'svm', load_classical('svm')
        elif choice == '3':
            return 'rf', load_classical('random_forest')
        elif choice == '4':
            return 'bert', load_bert()
        else:
            print('Введи число от 1 до 4')


def main():
    model_type, model_data = choose_model()

    print('\n' + '='*52)
    print('  Введи текст для классификации')
    print('  "сменить" — выбрать другую модель')
    print('  "выход"   — завершить')
    print('='*52 + '\n')

    while True:
        user_input = input('Текст: ').strip()

        if not user_input:
            continue

        if user_input.lower() == 'выход':
            print('До свидания!')
            break

        if user_input.lower() == 'сменить':
            model_type, model_data = choose_model()
            print()
            continue

        if len(user_input.split()) < 3:
            print('Введи текст подлиннее (минимум 3 слова)\n')
            continue

        if model_type == 'bert':
            result = predict_bert(user_input, *model_data)
        else:
            result = predict_classical(user_input, *model_data)

        print_result(result)

if __name__ == '__main__':
    main()