import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                            matthews_corrcoef, precision_score,
                            recall_score, confusion_matrix)
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

# Set the folder paths
fna_folder_path = os.path.join(os.getcwd(), 'input')
annotation_folder = os.path.join(os.getcwd(), 'annotation')

# Load the annotation CSV file
annotation_file = os.path.join(annotation_folder, 'annotation.csv')
if not os.path.exists(annotation_file):
    print(f"File not found at expected location: {annotation_file}")
    print("Please ensure the annotation.csv file is correctly placed in the data directory.")
    exit()

# Load the entire dataset
annotation_df = pd.read_csv(annotation_file)

# Reserve 10% of the data as an independent test set
train_val_df, test_df = train_test_split(annotation_df, test_size=0.1, random_state=42, stratify=annotation_df['label'])

# Prepare Stratified K-Fold cross-validation (5 folds)
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Load DNABERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")

# Create a dataset class
class DNADataset(Dataset):
    def __init__(self, file_paths, fna_folder_path, tokenizer, labels):
        self.file_paths = file_paths
        self.fna_folder_path = fna_folder_path
        self.tokenizer = tokenizer
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = os.path.join(self.fna_folder_path, self.file_paths[idx])
        sequence = self.read_fna_sequence(file_path)
        encodings = self.tokenize_sequence(sequence)
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def read_fna_sequence(self, file_path):
        with open(file_path, 'r') as file:
            sequence = ''
            for line in file:
                if line.startswith('>'):
                    if sequence:
                        return sequence
                    sequence = ''
                else:
                    sequence += line.strip()
            return sequence

    def tokenize_sequence(self, sequence):
        return self.tokenizer(sequence, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

# Function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    roc_auc = roc_auc_score(labels, preds, multi_class='ovr')
    mcc = matthews_corrcoef(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'eval_accuracy': acc,
        'eval_f1': f1,
        'eval_roc_auc': roc_auc,
        'eval_mcc': mcc,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_specificity': specificity
    }

# Cross-validation with hyperparameter tuning
best_accuracy = 0
best_model = None
best_params = None
results = []
metrics = {
    'fold': [],
    'accuracy': [],
    'f1': [],
    'roc_auc': [],
    'mcc': [],
    'precision': [],
    'recall': [],
    'specificity': []
}

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_df, train_val_df['label'])):
    print(f"Fold {fold + 1}/{num_folds}")

    # Create datasets for the current fold
    train_subset = train_val_df.iloc[train_idx]
    val_subset = train_val_df.iloc[val_idx]
    train_files = train_subset['filename'].tolist()
    train_labels = train_subset['label'].map({'S': 0, 'R': 1}).tolist()
    val_files = val_subset['filename'].tolist()
    val_labels = val_subset['label'].map({'S': 0, 'R': 1}).tolist()

    # Create datasets using DNADataset class
    train_dataset = DNADataset(train_files, fna_folder_path, tokenizer, train_labels)
    val_dataset = DNADataset(val_files, fna_folder_path, tokenizer, val_labels)

    # Load the DNABERT model
    model = BertForSequenceClassification.from_pretrained("zhihan1996/DNABERT-2-117M", num_labels=2).to(device)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f'./results/fold_{fold + 1}',
        num_train_epochs=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./logs/fold_{fold + 1}',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        greater_is_better=True,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the validation set
    val_metrics = trainer.evaluate()
    print(f"Validation metrics for fold {fold + 1}: {val_metrics}")

    # Store metrics for plotting later
    metrics['fold'].append(fold + 1)
    metrics['accuracy'].append(val_metrics['eval_accuracy'])
    metrics['f1'].append(val_metrics['eval_f1'])
    metrics['roc_auc'].append(val_metrics['eval_roc_auc'])
    metrics['mcc'].append(val_metrics['eval_mcc'])
    metrics['precision'].append(val_metrics['eval_precision'])
    metrics['recall'].append(val_metrics['eval_recall'])
    metrics['specificity'].append(val_metrics['eval_specificity'])

    if val_metrics['eval_accuracy'] > best_accuracy:
        best_accuracy = val_metrics['eval_accuracy']
        best_model = model
        best_params = {
            'num_epochs': 20,
            'batch_size': 8,
            'weight_decay': 0.01,
            'learning_rate': 2e-5
        }

# Evaluate on the independent test set
test_files = test_df['filename'].tolist()
test_labels = test_df['label'].map({'S': 0, 'R': 1}).tolist()
test_dataset = DNADataset(test_files, fna_folder_path, tokenizer, test_labels)

# Get predictions for the test set
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)  # Get the predicted classes
labels = predictions.label_ids  # True labels

# Compute metrics for the independent test set
independent_metrics = compute_metrics(predictions)

# Store independent test set metrics
metrics['fold'].append('Independent Set')
metrics['accuracy'].append(independent_metrics['eval_accuracy'])
metrics['f1'].append(independent_metrics['eval_f1'])
metrics['roc_auc'].append(independent_metrics['eval_roc_auc'])
metrics['mcc'].append(independent_metrics['eval_mcc'])
metrics['precision'].append(independent_metrics['eval_precision'])
metrics['recall'].append(independent_metrics['eval_recall'])
metrics['specificity'].append(independent_metrics['eval_specificity'])

# Print best parameters and accuracy
print(f"Best validation accuracy: {best_accuracy:.4f} with parameters: {best_params}")
print(f"Test metrics: {independent_metrics}")

# Save the best model and tokenizer
best_model.save_pretrained('./dnabert_amr_model')
tokenizer.save_pretrained('./dnabert_amr_model')

# Check for overlap between train and test sets
train_set = set(train_val_df['filename'].tolist())
test_set = set(test_files)
overlap = train_set.intersection(test_set)

if overlap:
    print(f"Data leakage detected! Overlapping files: {overlap}")
else:
    print("No data leakage detected.")

# Plotting the metrics as histograms
metrics_df = pd.DataFrame(metrics)

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Create a bar plot for each metric
x_labels = metrics_df['fold'].astype(str)  # Convert fold numbers to strings for x-axis labels
bar_width = 0.15  # Width of each bar
x = np.arange(len(x_labels))  # The label locations

# Loop through each metric and create a bar for it
for i, metric in enumerate(['accuracy', 'f1', 'roc_auc', 'mcc', 'precision', 'recall', 'specificity']):
    ax.bar(x + i * bar_width, metrics_df[metric], width=bar_width, label=metric)

# Set the x-ticks and labels
ax.set_xticks(x + bar_width * (len(['accuracy', 'f1', 'roc_auc', 'mcc', 'precision', 'recall', 'specificity']) - 1) / 2)

ax.set_xticklabels(x_labels)

ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Fold / Independent Set')
ax.set_ylabel('Score')
ax.set_title('Metrics per Fold and Independent Set')
ax.legend()
ax.grid()
plt.tight_layout()
plt.savefig('metrics_histogram.png')
plt.show()
