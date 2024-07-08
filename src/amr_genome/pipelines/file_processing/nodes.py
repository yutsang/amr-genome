#from src.amr_genome.pipelines.data_processing.my_imports import *
import pandas as pd
import io, tempfile, os, glob, sys
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from ftplib import FTP
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests, json, gzip, subprocess, multiprocessing
import shutil, itertools
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from kedro.pipeline import node
from typing import Tuple
from lazypredict.Supervised import LazyClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import run, CalledProcessError
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from typing import Dict, List
from lazypredict.Supervised import LazyClassifier
from sklearn.feature_extraction import DictVectorizer

pd.options.mode.chained_assignment = None

########
def process_file(args):
    filename, path_raw, count, total = args
    #print(f"Starting to process {filename}: [{count}/{total}]")
    prefix = filename.rstrip(".gz")
    output_path = f"{path_raw}_line/{prefix}"

    if os.path.isfile(output_path):
        #print(f"{filename} already processed. Skipped.")
        return

    try:
        if filename.endswith(".gz"):
            command = f"zcat '{path_raw}/{filename}' | sed ':a;N;/>/!s/\\n//;ta;P;D' > '{output_path}'"
        else:
            command = f"sed ':a;N;/>/!s/\\n//;ta;P;D' '{path_raw}/{filename}' > '{output_path}'"

        #print(f"Executing command: {command}")
        run(command, shell=True, check=True, timeout=30)
        #print(f"Finished processing {filename}")
    except CalledProcessError as e:
        print(f"Error processing {filename}: {e}")
    except Exception as e:
        print(f"Unexpected error processing {filename}: {e}")

def fna_line(path_raw: str) -> None:
    file_names = os.listdir(path_raw)
    total = len(file_names)
    print(f"Found {total} files in {path_raw}")

    os.makedirs(path_raw +"_line", exist_ok=True)

    # Prepare arguments for each file
    args_list = [(filename, path_raw, count, total) for count, filename in enumerate(file_names, start=1)]

    # Use Pool.map() to process files in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(process_file, args_list)

    print("All files processed")
    
#####
#Node: kmer_count:
def generate_kmer_hash(k, rpath_raw, filename, output_folder):
    k = str(k)
    prefix = filename.strip(".fna")
    filepath = os.path.join(rpath_raw, filename)
    command = f"jellyfish count -m {k} -s 1000000 -t 32 -o {os.path.join(output_folder, prefix)}.jf {filepath}"
    subprocess.run(command, shell=True)

def generate_kmer_hash_wrapper(args):
    return generate_kmer_hash(*args)

def process_kmer_counts(args):
    jf_file, query_path, output_folder = args
    prefix = os.path.basename(jf_file).strip(".jf")
    txt_file = os.path.join(output_folder, f"{prefix}.txt")
    
    # Dump k-mer counts
    dump_command = f"jellyfish dump {jf_file} > {txt_file}"
    subprocess.run(dump_command, shell=True)
    
    # Sort k-mer counts
    with open(query_path, 'r') as query_file:
        query_order = [line.strip() for line in query_file]

    kmer_counts = {}
    with open(txt_file, 'r') as input_file:
        for line in input_file:
            if line.startswith('>'):
                count = int(line[1:].strip())
            else:
                kmer = line.strip()
                kmer_counts[kmer] = count

    sorted_kmer_counts = []
    for kmer in query_order:
        count = kmer_counts.get(kmer, 0)
        sorted_kmer_counts.append((kmer, count))

    with open(txt_file, 'w') as output_file:
        for kmer, count in sorted_kmer_counts:
            output_file.write(f"{kmer}\t{count}\n")
    
    # Remove .jf file after processing
    os.remove(jf_file)
    
    return txt_file

def kmer_processing(k, rpath_raw, output_folder):
    def get_all_file_name(rpath_raw):
        return os.listdir(rpath_raw)

    file_names = get_all_file_name(rpath_raw)
    os.makedirs(output_folder, exist_ok=True)

    query_path = "./query.fasta"
    with open(query_path, 'w') as fd:
        for i in itertools.product('ATCG', repeat=int(k)):
            query = "".join(i)
            fd.write(f"{query}\n")

    # Generate k-mer hashes using pool map
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(generate_kmer_hash_wrapper, 
                        [(k, rpath_raw, filename, output_folder) for filename in file_names]), 
                total=len(file_names),
                desc="Generating k-mer hashes"))

    # Process k-mer counts using pool map
    jf_files = glob.glob(os.path.join(output_folder, '*.jf'))
    with Pool(cpu_count()) as p:
        txt_files = list(tqdm(p.imap(process_kmer_counts, 
                                    [(jf_file, query_path, output_folder) for jf_file in jf_files]),
                            total=len(jf_files),
                            desc="Processing k-mer counts"))

    return txt_files

#####
#Node: create_annotation
def create_annotation(stratified_sample: pd.DataFrame, path_kmer: str) -> pd.DataFrame:
    # Read input data
    df = stratified_sample.copy()
    
    # Get a set of valid genome IDs (files that exist in path_kmer)
    valid_genomes = set()
    for filename in os.listdir(path_kmer):
        if filename.endswith('.txt'):
            genome_id = filename.rsplit('.', 1)[0]  # Remove both .txt from the end
            valid_genomes.add(genome_id)
                
    # Filter rows based on valid genomes
    df['genbank_id_match'] = df['genbank_id'].astype(str).apply(lambda x: '.'.join(x.split('.')[:2]))
    print(df['genbank_id_match'].head())
    df = df[df['genbank_id_match'].isin(valid_genomes)]
    df = df.drop('genbank_id_match', axis=1)
    
    # Transform resistant_phenotype
    def transform_phenotype(phenotype):
        if phenotype == 'intermediate' or phenotype == 'resistant':
            return 1  # Set as binary label 1 for intermediate and resistant
        return 0  # Set as binary label 0 for susceptible
    
    df['resistant_phenotype'] = df['resistant_phenotype'].apply(transform_phenotype)
    
    # Select required columns
    columns = ['genome_id', 'genbank_id', 'tax_id', 'antibiotic', 'resistant_phenotype', 
            'class', 'antibiotic_class', 'taxonomy_label', 'strain_identifiers']
    df = df[columns]
    
    return df

#####
def stratified_split(annotation: pd.DataFrame, train_ratio: float, 
                    val_ratio: float, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Ensure the ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Select only required columns
    df = annotation[['genbank_id', 'resistant_phenotype']]
    
    # First split: separate test set
    train_val, test = train_test_split(
        df, 
        test_size=test_ratio, 
        stratify=df['resistant_phenotype'],
        random_state=42
    )
    
    # Second split: separate train and validation sets
    train_ratio_adjusted = train_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val, 
        train_size=train_ratio_adjusted, 
        stratify=train_val['resistant_phenotype'],
        random_state=42
    )
    
    return train, val, test
#####
#Node: merge_dfs

def merge_kmer_files(input_folder: str, output_file_path: str):
    """
    Merge k-mer count files into a single parquet file.
    
    Args:
    input_folder (str): Path to the folder containing k-mer count txt files.
    output_file (str): Path to save the output parquet file.
    """
    input_path = Path(input_folder)
    all_files = list(input_path.glob('*.txt'))
    
    # Initialize an empty DataFrame to store all k-mer counts
    df_all = pd.DataFrame()
    
    # Process each file
    for file in tqdm(all_files, desc="Processing files"):
        # Read the file
        df = pd.read_csv(file, sep='\t', header=None, names=['kmer', file.stem])
        
        # If this is the first file, use the k-mers as index
        if df_all.empty:
            df_all = df.set_index('kmer')
        else:
            # Merge with the existing DataFrame
            df_all = df_all.join(df.set_index('kmer'), how='outer')
    
    # Fill NaN values with 0
    df_all = df_all.fillna(0)
    
    # Convert all columns to int (except the index)
    for col in df_all.columns:
        df_all[col] = df_all[col].astype(int)
        
    df_all = df_all.reset_index()
    
    print(f"Merged data saved to {output_file_path}")
    return df_all

#####
#Node: Train
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yaml
import warnings

warnings.filterwarnings("ignore")

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train(kmer_df: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    # Load configuration
    config = load_config('./conf/config.yaml')

    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Exclude the 'kmer' column from normalization
    numeric_columns = kmer_df.columns.difference(['kmer'])

    # Normalize the numeric columns in kmer_data
    kmer_data_normalized = kmer_df.copy()
    kmer_data_normalized[numeric_columns] = scaler.fit_transform(kmer_data_normalized[numeric_columns])

    # Assuming kmer_data_normalized is your DataFrame
    X = kmer_data_normalized.iloc[:, 1:]  # Exclude the 'kmer' column
    y = annotation

    # Map filenames to labels
    #label_mapping = {'S': 0, 'R': 1}  # Assuming 'S' is susceptible and 'R' is resistant
    #y['label_encoded'] = y['resistant_phenotype'].map(label_mapping)
    y['label_encoded'] = y['resistant_phenotype']

    # Create a dictionary to map filenames to encoded labels
    label_dict = dict(zip(y['genbank_id'], y['label_encoded']))

    # Create labels array in the same order as X columns
    labels = [label_dict[col] for col in X.columns]

    # First split: 80% train+val, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X.T, labels, test_size=0.2, random_state=42)

    # Second split: 80% train, 20% val (64% train, 16% val, 20% test of total)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    class KmerDataset(Dataset):
        def __init__(self, dataframe, labels):
            self.data = torch.FloatTensor(dataframe.values)
            self.labels = torch.LongTensor(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    train_dataset = KmerDataset(X_train, y_train)
    val_dataset = KmerDataset(X_val, y_val)
    test_dataset = KmerDataset(X_test, y_test)

    batch_size = 32  # Adjust as needed

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the model
    class AMRPredictor(nn.Module):
        def __init__(self, input_size):
            super(AMRPredictor, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.sigmoid(self.fc3(x))
            return x

    # Initialize the model with the correct input size
    input_size = X_train.shape[1]  # Number of k-mers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AMRPredictor(input_size=input_size).to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['SOLVER']['SCHEDULER']['BASE_LR'])

    # Training loop
    num_epochs = config['SOLVER']['MAX_EPOCHS']
    for epoch in range(num_epochs):
        model.train()
        train_preds = []
        train_targets = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()
            train_preds.extend((outputs > 0.5).long().cpu().numpy())
            train_targets.extend(batch_y.cpu().numpy())
        train_accuracy = accuracy_score(train_targets, train_preds)
        train_precision = precision_score(train_targets, train_preds)
        train_recall = recall_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                val_preds.extend((outputs > 0.5).long().cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        val_accuracy = accuracy_score(val_targets, val_preds)
        val_precision = precision_score(val_targets, val_preds, average='binary')
        val_recall = recall_score(val_targets, val_preds, average='binary')

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train - Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}')
        print(f'Validation - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')

    # Evaluation
    model.eval()
    test_preds = []
    test_targets = []
    test_probs = []
    test_ids = []  # To store the IDs of test samples
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            test_probs.extend(outputs.cpu().numpy())
            test_preds.extend((outputs > 0.5).long().cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())
            test_ids.extend(X_test.index[len(test_preds)-len(outputs):len(test_preds)])

    # Create DataFrame with test results
    test_results = pd.DataFrame({
        'ID': test_ids,
        'Original': test_targets,
        'Prediction': test_preds,
        'Probability': test_probs
    })
    
    # Calculate and print overall metrics
    test_accuracy = accuracy_score(test_targets, test_preds)
    test_precision = precision_score(test_targets, test_preds, average='binary')
    test_recall = recall_score(test_targets, test_preds, average='binary')

    print(f'Test - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')

    return test_results


