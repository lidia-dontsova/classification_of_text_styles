# Веб-приложение для классификации стилей текста

from flask import Flask, request, jsonify, render_template
import joblib
import re
import torch
from pathlib import Path

app = Flask(__name__)

MODELS_DIR = Path(__file__).parent / 'models'

STYLES = {
    0: {'name': 'Научный',              'color': '#534AB7'},
    1: {'name': 'Официально-деловой',   'color': '#1D9E75'},
    2: {'name': 'Публицистический',     'color': '#185FA5'},
    3: {'name': 'Разговорный',          'color': '#D85A30'},
    4: {'name': 'Художественный',       'color': '#BA7517'},
}

STOPWORDS = {
    'и','в','не','на','что','я','с','он','как','это','по','но','от','к',
    'а','за','же','так','его','из','до','мне','они','у','была','был','было',
    'бы','есть','быть','при','уже','или','нет','нас','вот','для','вы','мы',
    'то','все','она','ее','её','ли','об','если','ещё','их','вас','вам',
    'им','тем','между','когда','там','потому','ним','нем','под','над','чтобы',
    'со','без','во','ты','тебя','тебе','ему','нам','них','ними',
}

# Кэш загруженных моделей
_cache = {}

def preprocess(text: str) -> str:
    text = re.sub(r'https?://\S+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^а-яёa-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) >= 3]
    return ' '.join(tokens)

def load_model(model_key: str):
    if model_key in _cache:
        return _cache[model_key]

    tfidf = joblib.load(MODELS_DIR / 'tfidf_vectorizer.pkl')

    if model_key == 'naive_bayes':
        model = joblib.load(MODELS_DIR / 'naive_bayes.pkl')
    elif model_key == 'svm':
        model = joblib.load(MODELS_DIR / 'svm.pkl')
    elif model_key == 'random_forest':
        model = joblib.load(MODELS_DIR / 'random_forest.pkl')
    elif model_key == 'rubert':
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        path      = MODELS_DIR / 'rubert_best'
        tokenizer = AutoTokenizer.from_pretrained(path)
        bert      = AutoModelForSequenceClassification.from_pretrained(path)
        bert.eval()
        _cache[model_key] = ('bert', tokenizer, bert)
        return _cache[model_key]

    _cache[model_key] = ('classical', tfidf, model)
    return _cache[model_key]

def predict(text: str, model_key: str) -> dict:
    data = load_model(model_key)

    if data[0] == 'bert':
        _, tokenizer, model = data
        inputs = tokenizer(
            text, max_length=256, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1)[0]
        pred_id   = probs.argmax().item()
        all_probs = {i: round(probs[i].item() * 100, 1) for i in range(5)}

    else:
        _, tfidf, model = data
        text_clean = preprocess(text)
        vec        = tfidf.transform([text_clean])
        pred_id    = int(model.predict(vec)[0])

        if hasattr(model, 'predict_proba'):
            probs     = model.predict_proba(vec)[0]
            all_probs = {i: round(float(probs[i]) * 100, 1) for i in range(5)}
        else:
            scores    = model.decision_function(vec)[0]
            exp_s     = [2 ** float(s) for s in scores]
            total     = sum(exp_s)
            all_probs = {i: round(exp_s[i] / total * 100, 1) for i in range(5)}

    return {
        'predicted_id':   pred_id,
        'predicted_name': STYLES[pred_id]['name'],
        'predicted_color': STYLES[pred_id]['color'],
        'confidence':     all_probs[pred_id],
        'all_probs':      [
            {
                'id':    i,
                'name':  STYLES[i]['name'],
                'color': STYLES[i]['color'],
                'prob':  all_probs[i],
            }
            for i in sorted(all_probs, key=lambda x: all_probs[x], reverse=True)
        ]
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data      = request.json
    text      = data.get('text', '').strip()
    model_key = data.get('model', 'svm')

    if not text or len(text.split()) < 3:
        return jsonify({'error': 'Введите текст (минимум 3 слова)'}), 400

    try:
        result = predict(text, model_key)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
