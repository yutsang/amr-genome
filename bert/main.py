from data_processing import DataLoader
from model_training import train_model
from utils import check_saved_model

if __name__ == "__main__":
    raw_data_directory = './input/raw_noline'
    model_save_directory = './model/spm'
    
    # Step 1: Initialize DataLoader and train SentencePiece models
    data_loader = DataLoader(raw_data_directory, model_save_directory)
    
    best_model_name = data_loader.train_sentencepiece_models()
    
    # Step 2: Train the BERT model using the best SentencePiece model found
    data_folder = './input/raw_spm/'
    spm_model_path = f'{model_save_directory}/{best_model_name}.model'
    
    train_model(data_folder, spm_model_path)

    # Step 3: Check saved models after training
    check_saved_model()
