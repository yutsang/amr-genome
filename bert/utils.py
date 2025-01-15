import os

def check_saved_model(model_dir='./results'):
    try:
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.bin') or f.endswith('.pt')]
        print("Saved model files:", model_files)
    except FileNotFoundError:
        print(f"Directory '{model_dir}' does not exist.")
