# Fine-tuning RuBERT для классификации стилей

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# конфигурация
BASE      = Path('/Users/lidia.donts/Documents/учеба/python_projects/text-styles/cmd/datasets')
OUT_DIR   = Path('/Users/lidia.donts/Documents/учеба/python_projects/text-styles/cmd/models')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME  = 'DeepPavlov/rubert-base-cased'
MAX_LEN     = 256
BATCH_SIZE  = 16
EPOCHS      = 4
LR          = 2e-5
NUM_CLASSES = 5
RANDOM_SEED = 42

STYLES = {
    0: 'Научный',
    1: 'Офиц.-деловой',
    2: 'Публицистический',
    3: 'Разговорный',
    4: 'Художественный',
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# загрузка данных
train = pd.read_csv(BASE / 'train.csv')
val   = pd.read_csv(BASE / 'val.csv')
test  = pd.read_csv(BASE / 'test.csv')

print(f'Train: {len(train)} | Val: {len(val)} | Test: {len(test)}')

# датасет
class StyleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts.tolist()
        self.labels    = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

# загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)
model = model.to(device)

train_dataset = StyleDataset(train['text_ready'], train['label'], tokenizer, MAX_LEN)
val_dataset   = StyleDataset(val['text_ready'],   val['label'],   tokenizer, MAX_LEN)
test_dataset  = StyleDataset(test['text_ready'],  test['label'],  tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# оптимизатор и планировщик
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps   = len(train_loader) * EPOCHS
warmup_steps  = total_steps // 10

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# функции обучения и оценки
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds   = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1  = f1_score(all_labels, all_preds, average='macro')
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return f1, acc, all_preds, all_labels

# обучение
best_val_f1   = 0
best_epoch    = 0
history       = []

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_f1, val_acc, _, _ = evaluate_model(model, val_loader, device)

    history.append({
        'epoch': epoch,
        'train_loss': round(train_loss, 4),
        'train_acc':  round(train_acc, 4),
        'val_f1':     round(val_f1, 4),
        'val_acc':    round(val_acc, 4),
    })

    print(f'Эпоха {epoch}/{EPOCHS} | '
          f'Loss: {train_loss:.4f} | '
          f'Train Acc: {train_acc:.4f} | '
          f'Val F1: {val_f1:.4f} | '
          f'Val Acc: {val_acc:.4f}')

    # Сохраняем лучшую модель по Val F1
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch  = epoch
        model.save_pretrained(OUT_DIR / 'rubert_best')
        tokenizer.save_pretrained(OUT_DIR / 'rubert_best')
        print(f'Лучшая модель сохранена (Val F1={val_f1:.4f})')

print(f'\nЛучшая эпоха: {best_epoch} | Val F1: {best_val_f1:.4f}')

# финальная оценка на test
best_model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR / 'rubert_best')
best_model = best_model.to(device)

test_f1, test_acc, test_preds, test_labels = evaluate_model(best_model, test_loader, device)

print(f'\n{"="*50}')
print(f'  RuBERT — финальные результаты (test)')
print(f'{"="*50}')
print(f'Test F1-macro:  {test_f1:.4f}')
print(f'Test Accuracy:  {test_acc:.4f}')
print(f'\nClassification Report:')
print(classification_report(
    test_labels, test_preds,
    target_names=list(STYLES.values())
))

print('Confusion Matrix:')
cm = confusion_matrix(test_labels, test_preds)
cm_df = pd.DataFrame(cm, index=STYLES.values(), columns=STYLES.values())
print(cm_df.to_string())

# сохранение результатов
pd.DataFrame(history).to_csv(OUT_DIR / 'bert_history.csv', index=False, encoding='utf-8-sig')

bert_result = pd.DataFrame([{
    'model':             'RuBERT',
    'val_f1_macro':      round(best_val_f1, 4),
    'test_f1_macro':     round(test_f1, 4),
    'test_f1_weighted':  round(f1_score(test_labels, test_preds, average='weighted'), 4),
    'test_accuracy':     round(test_acc, 4),
}])

# сохранение результатов BERT
bert_result.to_csv(OUT_DIR / 'bert_results.csv', index=False)
print(bert_result.to_string(index=False))
