"""
Classification de Sentiment avec DistilBERT
Analyse de critiques de films IMDb (positif/négatif)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TRAIN_SIZE = 1000  # Réduit pour rapidité (max 25000)
    TEST_SIZE = 500    # Réduit pour rapidité (max 25000)

print(f"Utilisation de: {Config.DEVICE}")

# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
print("\n=== Chargement des données IMDb ===")
dataset = load_dataset('imdb')

# Réduction du dataset pour accélérer l'entraînement
train_data = dataset['train'].shuffle(seed=42).select(range(Config.TRAIN_SIZE))
test_data = dataset['test'].shuffle(seed=42).select(range(Config.TEST_SIZE))

print(f"Taille train: {len(train_data)}")
print(f"Taille test: {len(test_data)}")

# Exemple de données
print("\nExemple de critique:")
print(f"Texte: {train_data[0]['text'][:200]}...")
print(f"Label: {'Positif' if train_data[0]['label'] == 1 else 'Négatif'}")

# 2. TOKENIZATION
print("\n=== Tokenization avec DistilBERT ===")
tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)

class IMDbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Création des datasets
train_dataset = IMDbDataset(
    train_data['text'], 
    train_data['label'], 
    tokenizer, 
    Config.MAX_LENGTH
)

test_dataset = IMDbDataset(
    test_data['text'], 
    test_data['label'], 
    tokenizer, 
    Config.MAX_LENGTH
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

# 3. MODÈLE
print("\n=== Chargement du modèle DistilBERT ===")
model = DistilBertForSequenceClassification.from_pretrained(
    Config.MODEL_NAME,
    num_labels=2
)
model = model.to(Config.DEVICE)

# 4. ENTRAÎNEMENT
print("\n=== Configuration de l'entraînement ===")
optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
total_steps = len(train_loader) * Config.EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    correct_predictions = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return (correct_predictions.double() / len(data_loader.dataset), 
            np.mean(losses), 
            predictions, 
            true_labels)

# Entraînement
print("\n=== Entraînement du modèle ===")
history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}

for epoch in range(Config.EPOCHS):
    print(f'\nEpoch {epoch + 1}/{Config.EPOCHS}')
    
    train_acc, train_loss = train_epoch(
        model, 
        train_loader, 
        optimizer, 
        scheduler, 
        Config.DEVICE
    )
    
    val_acc, val_loss, _, _ = eval_model(model, test_loader, Config.DEVICE)
    
    history['train_acc'].append(train_acc.item())
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc.item())
    history['val_loss'].append(val_loss)
    
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

# 5. ÉVALUATION FINALE
print("\n=== Évaluation Finale ===")
_, _, y_pred, y_true = eval_model(model, test_loader, Config.DEVICE)

print("\nRapport de classification:")
print(classification_report(y_true, y_pred, target_names=['Négatif', 'Positif']))

# 6. VISUALISATIONS
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Courbes d'apprentissage
axes[0].plot(history['train_acc'], label='Train Accuracy')
axes[0].plot(history['val_acc'], label='Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Courbes d\'Apprentissage')
axes[0].legend()
axes[0].grid(True)

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Négatif', 'Positif'],
            yticklabels=['Négatif', 'Positif'])
axes[1].set_xlabel('Prédiction')
axes[1].set_ylabel('Vérité')
axes[1].set_title('Matrice de Confusion')

plt.tight_layout()
plt.savefig('resultats_sentiment_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAccuracy finale: {accuracy_score(y_true, y_pred):.4f}")

# 7. TEST SUR DE NOUVEAUX EXEMPLES
print("\n=== Test sur de nouveaux exemples ===")

def predict_sentiment(text, model, tokenizer, device, max_length=256):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, pred = torch.max(outputs.logits, dim=1)
    
    return 'Positif' if pred.item() == 1 else 'Négatif'

exemples = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "Terrible film, waste of time. The plot was boring and predictable.",
    "An interesting concept but poorly executed. Mixed feelings overall."
]

for texte in exemples:
    sentiment = predict_sentiment(texte, model, tokenizer, Config.DEVICE)
    print(f"\nTexte: {texte}")
    print(f"Sentiment prédit: {sentiment}")

print("\n✓ Entraînement terminé ! Le graphique a été sauvegardé.")
