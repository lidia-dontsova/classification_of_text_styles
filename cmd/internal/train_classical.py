# Обучение классических моделей на TF-IDF признаках

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib

BASE     = Path('/Users/lidia.donts/Documents/учеба/python_projects/text-styles/cmd/datasets')
OUT_DIR  = Path('/Users/lidia.donts/Documents/учеба/python_projects/text-styles/cmd/models')
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLES = {
    0: 'Научный',
    1: 'Офиц.-деловой',
    2: 'Публицистический',
    3: 'Разговорный',
    4: 'Художественный',
}

# загрузка данных
train = pd.read_csv(BASE / 'train.csv')
val   = pd.read_csv(BASE / 'val.csv')
test  = pd.read_csv(BASE / 'test.csv')

X_train, y_train = train['text_ready'].astype(str), train['label']
X_val,   y_val   = val['text_ready'].astype(str),   val['label']
X_test,  y_test  = test['text_ready'].astype(str),  test['label']

print(f'Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}')

# TF-IDF векторизатор
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.90,
    sublinear_tf=True,
    max_features=50_000,
)

# обучение только на train
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(X_test)

print(f'TF-IDF матрица: {X_train_tfidf.shape}')


# фунция оценки модели
def evaluate(name, model, X_val, y_val, X_test, y_test):
    print(f'\n{"="*50}')
    print(f'  {name}')
    print(f'{"="*50}')

    # Оценка на val
    y_val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    print(f'Val F1-macro: {val_f1:.4f}')

    # Финальная оценка на test
    y_test_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    print(f'Test F1-macro: {test_f1:.4f}')

    print(f'\nClassification Report (test):')
    print(classification_report(
        y_test, y_test_pred,
        target_names=list(STYLES.values())
    ))

    print(f'Confusion Matrix (test):')
    cm = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm, index=STYLES.values(), columns=STYLES.values())
    print(cm_df.to_string())

    return {
        'model': name,
        'val_f1_macro': round(val_f1, 4),
        'test_f1_macro': round(test_f1, 4),
        'test_f1_weighted': round(f1_score(y_test, y_test_pred, average='weighted'), 4),
        'test_accuracy': round((y_test == y_test_pred).mean(), 4),
    }

# Naive Bayes
nb_params = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]}
nb_grid = GridSearchCV(
    MultinomialNB(),
    nb_params,
    cv=StratifiedKFold(5),
    scoring='f1_macro',
    n_jobs=-1
)
nb_grid.fit(X_train_tfidf, y_train)
print(f'Лучший alpha: {nb_grid.best_params_["alpha"]}')
nb_best = nb_grid.best_estimator_
joblib.dump(nb_best, OUT_DIR / 'naive_bayes.pkl')

nb_result = evaluate('Naive Bayes', nb_best, X_val_tfidf, y_val, X_test_tfidf, y_test)

# SVM 
svm_params = {'C': [0.1, 1.0, 5.0, 10.0]}
svm_grid = GridSearchCV(
    LinearSVC(max_iter=2000, class_weight='balanced', dual=True),
    svm_params,
    cv=StratifiedKFold(5),
    scoring='f1_macro',
    n_jobs=-1
)
svm_grid.fit(X_train_tfidf, y_train)
print(f'Лучший C: {svm_grid.best_params_["C"]}')
svm_best = svm_grid.best_estimator_
joblib.dump(svm_best, OUT_DIR / 'svm.pkl')

svm_result = evaluate('SVM (LinearSVC)', svm_best, X_val_tfidf, y_val, X_test_tfidf, y_test)

# Random Forest
rf_params = {
    'n_estimators': [100, 300],
    'max_depth':    [None, 30],
    'min_samples_leaf': [1, 3],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    rf_params,
    cv=StratifiedKFold(5),
    scoring='f1_macro',
    n_jobs=-1
)
rf_grid.fit(X_train_tfidf, y_train)
print(f'Лучшие параметры: {rf_grid.best_params_}')
rf_best = rf_grid.best_estimator_
joblib.dump(rf_best, OUT_DIR / 'random_forest.pkl')

rf_result = evaluate('Random Forest', rf_best, X_val_tfidf, y_val, X_test_tfidf, y_test)

# СРАВНИТЕЛЬНАЯ ТАБЛИЦА
results = pd.DataFrame([nb_result, svm_result, rf_result])
print('  СРАВНИТЕЛЬНАЯ ТАБЛИЦА')
print(results.to_string(index=False))

results.to_csv(OUT_DIR / 'classical_results.csv', index=False, encoding='utf-8-sig')

# Сохраняем векторизатор 
joblib.dump(tfidf, OUT_DIR / 'tfidf_vectorizer.pkl')
