import os
import random
import sentencepiece as spm
from tqdm import tqdm

class DataLoader:
    def __init__(self, raw_data_directory, model_save_directory, spm_size=4096):
        self.raw_data_directory = raw_data_directory
        self.model_save_directory = model_save_directory
        self.spm_size = spm_size
        self.fna_files = self.load_fna_files()

    def load_fna_files(self):
        if os.path.exists(self.raw_data_directory):
            return [os.path.join(self.raw_data_directory, f) for f in os.listdir(self.raw_data_directory) if f.endswith('.fna')]
        return []

    def train_sentencepiece_models(self, trials=10):
        best_model = None
        best_perplexity = float('inf')

        for trial in range(trials):
            subset_size = min(5, int(0.05 * len(self.fna_files)))
            selected_files = random.sample(self.fna_files, subset_size)
            text_file_path = os.path.join(self.model_save_directory, f'trial_{trial + 1}_combined_sequences.txt')

            self.prepare_training_file(selected_files, text_file_path)
            spm_model_prefix = os.path.join(self.model_save_directory, f'spm_model_trial_{trial + 1}')
            spm.SentencePieceTrainer.Train(f'--input={text_file_path} --model_prefix={spm_model_prefix} --vocab_size={self.spm_size}')

            perplexity = self.evaluate_model(spm_model_prefix, selected_files)
            print(f"Trial {trial + 1}: Average pieces per character = {perplexity:.2f}")

            if perplexity < best_perplexity:
                best_perplexity = perplexity
                best_model = f'spm_model_trial_{trial + 1}'

        return best_model

    def prepare_training_file(self, selected_files, text_file_path):
        with open(text_file_path, 'w') as outfile:
            for fna_file in selected_files:
                with open(fna_file, 'r') as infile:
                    outfile.write(infile.read() + '\n')

    def evaluate_model(self, spm_model_prefix, selected_files):
        sp = spm.SentencePieceProcessor(model_file=f'{spm_model_prefix}.model')
        total_length = total_pieces = 0

        for fna_file in selected_files:
            with open(fna_file, 'r') as infile:
                for line in infile:
                    if line.startswith('>'):
                        continue
                    encoded_line = sp.encode(line.strip(), out_type=str)
                    total_length += len(line.strip())
                    total_pieces += len(encoded_line)

        return total_pieces / total_length if total_length > 0 else float('inf')
