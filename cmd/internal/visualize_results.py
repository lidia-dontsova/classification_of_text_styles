# Визуализация результатов 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

BASE    = Path('/Users/lidia.donts/Documents/учеба/python_projects/text-styles/cmd')
OUT_DIR = BASE / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family']   = 'DejaVu Sans'
plt.rcParams['figure.dpi']    = 150
plt.rcParams['savefig.dpi']   = 300
plt.rcParams['savefig.bbox']  = 'tight'

STYLES = ['Научный', 'Офиц.-деловой', 'Публицистич.', 'Разговорный', 'Художествен.']
MODELS = ['Naive Bayes', 'Random Forest', 'SVM', 'RuBERT']

# СРАВНИТЕЛЬНАЯ ТАБЛИЦА МОДЕЛЕЙ 
results = {
    'Модель':          MODELS,
    'Val F1-macro':    [0.9522, 0.9347, 0.9644, 0.9886],
    'Test F1-macro':   [0.9523, 0.9308, 0.9719, 0.9899],
    'Test Accuracy':   [0.9524, 0.9310, 0.9719, 0.9900],
}
df_results = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(MODELS))
w = 0.25

bars1 = ax.bar(x - w, df_results['Val F1-macro'],  w, label='Val F1-macro',  color='#534AB7', alpha=0.85)
bars2 = ax.bar(x,     df_results['Test F1-macro'], w, label='Test F1-macro', color='#1D9E75', alpha=0.85)
bars3 = ax.bar(x + w, df_results['Test Accuracy'], w, label='Accuracy',      color='#D85A30', alpha=0.85)

# Подписи значений
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=11)
ax.set_ylim(0.90, 1.01)
ax.set_ylabel('Значение метрики', fontsize=11)
ax.set_title('Сравнение моделей классификации стилей текста', fontsize=13, pad=15)
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax.grid(axis='y', alpha=0.3)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / '1_model_comparison.png')
plt.close()

# CONFUSION MATRICES 
cm_nb = np.array([
    [290,   2,   2,   0,   6],
    [  0, 297,   0,   0,   0],
    [  4,   2, 277,  16,   0],
    [  2,   0,  15, 267,  14],
    [  2,   0,   5,   1, 291],
])

cm_svm = np.array([
    [294,   2,   2,   0,   2],
    [  0, 297,   0,   0,   0],
    [  1,   1, 289,   8,   0],
    [  0,   1,   4, 284,   9],
    [  3,   0,   1,   8, 287],
])

cm_rf = np.array([
    [285,   3,   4,   3,   5],
    [  1, 296,   0,   0,   0],
    [  5,   1, 282,  11,   0],
    [ 16,   0,   6, 256,  20],
    [  7,   0,   3,  18, 271],
])

cm_bert = np.array([
    [297,   1,   1,   0,   1],
    [  0, 297,   0,   0,   0],
    [  0,   0, 298,   1,   0],
    [  0,   0,   3, 289,   6],
    [  0,   0,   0,   2, 297],
])

cms    = [cm_nb, cm_rf, cm_svm, cm_bert]
titles = ['Naive Bayes', 'Random Forest', 'SVM (LinearSVC)', 'RuBERT']

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for i, (cm, title) in enumerate(zip(cms, titles)):
    # Нормализуем по строкам (recall)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=cm, fmt='d',
        cmap='Blues', ax=axes[i],
        xticklabels=STYLES, yticklabels=STYLES,
        vmin=0, vmax=1,
        linewidths=0.5, linecolor='white',
        cbar_kws={'format': '%.1f'}
    )
    axes[i].set_title(f'{title}  (F1={df_results["Test F1-macro"].iloc[i if i<3 else 3]:.4f})',
                      fontsize=12, pad=10)
    axes[i].set_xlabel('Предсказанный класс', fontsize=10)
    axes[i].set_ylabel('Истинный класс', fontsize=10)
    axes[i].tick_params(axis='x', rotation=30, labelsize=9)
    axes[i].tick_params(axis='y', rotation=0,  labelsize=9)

plt.suptitle('Матрицы ошибок (confusion matrix) — все модели\nЦифры = количество текстов, цвет = доля правильных',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / '2_confusion_matrices.png')
plt.close()

# КРИВАЯ ОБУЧЕНИЯ BERT
history = pd.read_csv(BASE / 'models/bert_history.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss
ax1.plot(history['epoch'], history['train_loss'],
         'o-', color='#534AB7', label='Train Loss', linewidth=2)
ax1.set_xlabel('Эпоха', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Функция потерь (Train Loss)', fontsize=12)
ax1.set_xticks(history['epoch'])
ax1.grid(alpha=0.3)
ax1.spines[['top','right']].set_visible(False)
ax1.legend(fontsize=10)

# F1 Val
ax2.plot(history['epoch'], history['val_f1'],
         's-', color='#1D9E75', label='Val F1-macro', linewidth=2)
ax2.plot(history['epoch'], history['val_acc'],
         '^-', color='#D85A30', label='Val Accuracy', linewidth=2, linestyle='--')
ax2.set_xlabel('Эпоха', fontsize=11)
ax2.set_ylabel('Значение метрики', fontsize=11)
ax2.set_title('Качество на Val выборке', fontsize=12)
ax2.set_xticks(history['epoch'])
ax2.set_ylim(0.93, 1.0)
ax2.grid(alpha=0.3)
ax2.spines[['top','right']].set_visible(False)
ax2.legend(fontsize=10)

plt.suptitle('Кривая обучения RuBERT (fine-tuning, 4 эпохи)', fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / '3_bert_learning_curve.png')
plt.close()

# F1 ПО КЛАССАМ 
f1_by_class = pd.DataFrame({
    'Стиль':         STYLES,
    'Naive Bayes':   [0.97, 0.99, 0.93, 0.92, 0.95],
    'Random Forest': [0.93, 0.99, 0.95, 0.87, 0.91],
    'SVM':           [0.98, 0.99, 0.97, 0.95, 0.96],
    'RuBERT':        [0.99, 1.00, 0.99, 0.98, 0.99],  # из твоего Classification Report
})

fig, ax = plt.subplots(figsize=(11, 5))
x   = np.arange(len(STYLES))
w   = 0.18
colors = ['#534AB7', '#D85A30', '#1D9E75', '#BA7517']

for i, (model, color) in enumerate(zip(MODELS, colors)):
    offset = (i - 1.5) * w
    bars = ax.bar(x + offset, f1_by_class[model], w,
                  label=model, color=color, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(STYLES, fontsize=11)
ax.set_ylim(0.84, 1.02)
ax.set_ylabel('F1-score', fontsize=11)
ax.set_title('F1-score по стилям — сравнение всех моделей', fontsize=13, pad=15)
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='y', alpha=0.3)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / '4_f1_by_class.png')
plt.close()

print('  1_model_comparison.png  — сравнение моделей')
print('  2_confusion_matrices.png — матрицы ошибок')
print('  3_bert_learning_curve.png — кривая обучения BERT')
print('  4_f1_by_class.png        — F1 по стилям')