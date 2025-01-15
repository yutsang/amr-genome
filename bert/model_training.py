import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForPreTraining, BertConfig, Trainer, TrainingArguments
import sentencepiece as spm
from tqdm import tqdm

class CustomTokenizer:
    def __init__(self, model_path):
        self.tokenizer = spm.SentencePieceProcessor(model_file=model_path)

    def __call__(self, text):
        ids = self.tokenizer.EncodeAsIds(text)
        return {
            'input_ids': torch.tensor(ids[:25]),
            'attention_mask': torch.tensor([1] * min(len(ids), 25)),
            'token_type_ids': torch.tensor([0] * min(len(ids), 25))
        }

class FnaDataset(Dataset):
    def __init__(self, file_paths, tokenizer):
        self.tokenizer = tokenizer
        self.data = self.prepare_dataset(file_paths)

    def prepare_dataset(self, file_paths):
        data_list = []
        for file_path in tqdm(file_paths, desc="Processing files"):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for i in range(len(lines) - 1):
                    sentence_a = lines[i].strip()
                    sentence_b = lines[i + 1].strip()
                    tokenized_a = self.tokenizer(sentence_a)
                    tokenized_b = self.tokenizer(sentence_b)

                    if tokenized_a['input_ids'].size(0) == 25 and tokenized_b['input_ids'].size(0) == 25:
                        input_ids = torch.cat((tokenized_a['input_ids'], tokenized_b['input_ids']))
                        attention_mask = torch.cat((tokenized_a['attention_mask'], tokenized_b['attention_mask']))
                        token_type_ids = torch.tensor([0] * len(tokenized_a['input_ids']) + [1] * len(tokenized_b['input_ids']))
                        labels = input_ids.clone()
                        mask_indices = torch.randint(0, len(input_ids), (5,))
                        labels[mask_indices] = input_ids[mask_indices]
                        labels[~labels.eq(input_ids)] = -100
                        nsp_label = torch.tensor(1)

                        data_list.append({
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'token_type_ids': token_type_ids,
                            'labels': labels,
                            'next_sentence_label': nsp_label
                        })
        return data_list

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def create_model(vocab_size):
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
    )
    return BertForPreTraining(config)

def train_model(data_folder, spm_model_path):
    if not os.path.exists(spm_model_path):
        print(f"Error: SentencePiece model not found at: {spm_model_path}")
        return

    fna_files = glob.glob(os.path.join(data_folder, '*.fna'))
    tokenizer = CustomTokenizer(model_path=spm_model_path)
    dataset = FnaDataset(fna_files, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = create_model(vocab_size=4096)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model()
