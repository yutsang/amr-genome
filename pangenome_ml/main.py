import os
import pandas as pd
from data_processing import process_excel_file
from model_training import train_and_evaluate_models

def main():
    # Process Excel file and obtain data
    metadata_df, pangenome_transposed, merged_df, target_cols = process_excel_file()

    # Train and evaluate models for each target
    train_and_evaluate_models(merged_df, target_cols)

if __name__ == "__main__":
    main()
