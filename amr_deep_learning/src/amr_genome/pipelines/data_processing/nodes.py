#from src.amr_genome.pipelines.data_processing.my_imports import *
import pandas as pd
import io, tempfile, os, glob
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import StratifiedShuffleSplit
from ftplib import FTP
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests, json, gzip, subprocess, multiprocessing
import shutil
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from kedro.pipeline import node
from sklearn.model_selection import train_test_split
from typing import Tuple
from lazypredict.Supervised import LazyClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from kedro_datasets.pandas import ExcelDataset

pd.options.mode.chained_assignment = None

def load_process_sample(df_antibiotic_metadata_path: str, antibiotic_class_json_path: str, test_size: float, antibiotic: str) -> pd.DataFrame:
    # Load data
    df_amr = pd.read_excel(df_antibiotic_metadata_path, sheet_name="AMR_data")
    df_genome = pd.read_excel(df_antibiotic_metadata_path, sheet_name='genome_metadata')
    
    # Filter data if necessary
    if antibiotic != 'all':
        df_amr = df_amr[df_amr['antibiotic'] == antibiotic]
    
    # Merge data
    merge = pd.merge(df_genome, df_amr, on='genome_id').dropna(subset=["genbank_id"])
    print('merge', merge.shape)

    # Process data
    with open(antibiotic_class_json_path, 'r') as file:
        data = json.load(file)
    merge['antibiotic_class'] = merge['antibiotic'].map(data['antibiotic_classes'])
    merge['taxonomy_label'] = merge['taxonomy'].str.split(' ', n=2).str[:2].str.join(' ')
    merge['strain_identifiers'] = merge['taxonomy'].str.split(' ', n=2).str[2]
    merge = merge.drop(columns=['taxonomy'])
    
    # Replace rare taxonomy labels and resistant phenotype
    merge['taxonomy_label'] = merge['taxonomy_label'].map(data['taxonomy_genus']).fillna('Other')
    merge['resistant_phenotype'] = merge['resistant_phenotype'].replace('intermediate', 'resistant')

    # Stratified sampling
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for _, sample_index in sss.split(merge, merge['resistant_phenotype']):
        stratified_sample = merge.iloc[sample_index]

    return stratified_sample

def ftp_connect(host, dir_path):
    ftp = FTP(host)
    ftp.login()
    ftp.cwd(dir_path)
    return ftp

def get_links(ftp, genbank_id, current_dir, links):
    files = []
    ftp.dir(files.append)
    for file in files:
        words = file.split(maxsplit=8)
        filename = words[-1]
        if filename not in ['.', '..']:
            if file.startswith('d'):  # if it is a directory
                ftp.cwd(filename)
                get_links(ftp, genbank_id, current_dir + filename + '/', links)
                ftp.cwd('..')
            elif filename.endswith('_genomic.fna.gz'):  # if it is the target file
                link = 'https://ftp.ncbi.nlm.nih.gov' + current_dir + filename
                links.append(link)

def get_ftp_link(genbank_id):
    host = 'ftp.ncbi.nlm.nih.gov'
    split_id = genbank_id.split('_')[1][:-2]  # split the id
    dir_path = '/genomes/all/' + genbank_id.split('_')[0] + '/' + '/'.join([split_id[i:i+3] for i in range(0, len(split_id), 3)]) + '/'
    ftp = ftp_connect(host, dir_path)
    links = []
    get_links(ftp, genbank_id, dir_path, links)
    return links[0] if links else None

def get_and_set_link(df: pd.DataFrame, max_workers: int) -> pd.DataFrame:
    # Check if 'merge_ftp_df.csv' exists in '/data/02_intermediate/', if yes delete it
    file_path = '/data/02_intermediate/merge_ftp_df.csv'
    if os.path.isfile(file_path):
        os.remove(file_path)

    # Check if 'ftp_link' column exists, if not create it
    if 'ftp_link' not in df.columns:
        df['ftp_link'] = None

    # Filter dataframe to only rows where 'ftp_link' is not already set
    df_to_update = df[df['ftp_link'].isnull()]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_ftp_link, df_to_update.at[i, 'genbank_id']): i for i in df_to_update.index}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching links"):
            df.at[futures[future], 'ftp_link'] = future.result()
    
    return df

def download_and_unzip(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data/02_intermediate', 'fna')
    os.makedirs(data_dir, exist_ok=True)

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[ 500, 502, 503, 504 ])
    session.mount('http://', HTTPAdapter(max_retries=retries))

    pbar = tqdm(total=df.shape[0])

    def _download_and_unzip(row):
        new_file_name = str(row['genbank_id']) + '.fna.gz'
        unzipped_file_name = new_file_name[:-3]

        if os.path.exists(os.path.join(data_dir, unzipped_file_name)):
            pbar.update()
            return

        ftp_link = row['ftp_link']
        response = session.get(ftp_link, stream=True)
        original_file_name = ftp_link.split('/')[-1]

        with open(os.path.join(data_dir, original_file_name), 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        os.rename(os.path.join(data_dir, original_file_name), os.path.join(data_dir, new_file_name))

        with gzip.open(os.path.join(data_dir, new_file_name), 'rb') as f_in:
            with open(os.path.join(data_dir, unzipped_file_name), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        if os.path.exists(os.path.join(data_dir, new_file_name)):
            os.remove(os.path.join(data_dir, new_file_name))

        pbar.update()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(_download_and_unzip, df.to_dict('records'))
    pbar.close()

#####
'''
def generate_all_kmers(k):
    bases = ['A', 'C', 'G', 'T']
    return [''.join(p) for p in itertools.product(bases, repeat=k)]

def count_kmers(sequence, k):
    kmer_freq = {}
    for i in range(len(sequence) - int(k) + 1):
        kmer = sequence[i:i+k]
        if kmer in kmer_freq:
            kmer_freq[kmer] += 1
        else:
            kmer_freq[kmer] = 1
    return kmer_freq

def process_fna(file_path, k, max_workers):
    #print(f"Starting to process {file_path}")
    sequences = []
    with open(file_path, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            sequences.append(str(record.seq))

    kmer_freqs = []
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(count_kmers, sequence, k) for sequence in sequences]
        for future in futures:
            kmer_freqs.append(future.result())

    df = pd.DataFrame(kmer_freqs).fillna(0).sum().astype(int).to_frame().transpose()
    df['file'] = file_path
    #print(f"Finished processing {file_path}")
    return df

def process_files(k, max_workers):
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data/02_intermediate', 'fna')
    file_paths = glob.glob(data_dir+"/*.fna")
    dfs = []

    with ThreadPoolExecutor(max_workers) as executor:
        futures = {executor.submit(process_fna, file_path, k, max_workers): file_path for file_path in file_paths}
        for future in tqdm(as_completed(futures.keys()), total=len(futures), desc="Processing Files"):
            file_path = futures[future]
            try:
                df = future.result()
                dfs.append(df)
                #print(f"Finished processing {file_path}")
            except Exception as e:
                print(f"An exception occurred while processing {file_path}: {e}")

    df_all_kmers = pd.concat(dfs, ignore_index=True)
    df_all_kmers.fillna(0, inplace=True)
    df_all_kmers = df_all_kmers.astype({col: int for col in df_all_kmers.columns if col != 'file'})
    df_all_kmers = df_all_kmers.groupby('file').sum().reset_index()

    missing_files = set(file_paths) - set(df_all_kmers['file'])
    empty_rows = pd.DataFrame(0, columns=df_all_kmers.columns.drop('file'), index=range(len(missing_files)))
    empty_rows['file'] = list(missing_files)
    df_all_kmers = pd.concat([df_all_kmers, empty_rows], ignore_index=True)

    return df_all_kmers

##
def merge_dfs(merge_ftp_df: pd.DataFrame, genomes_kmers: pd.DataFrame) -> pd.DataFrame:
    # Extract the GenBank ID from the file path
    genomes_kmers['genbank_id'] = genomes_kmers['file'].str.extract(r'(GCA_\d+\.\d+)')

    # Merge the dataframes on the genbank_id column
    merged_df = pd.merge(merge_ftp_df, genomes_kmers, on='genbank_id')

    return merged_df

## Data Preparation for lazypredict

def prepare_data(df: pd.DataFrame, k: int) -> pd.DataFrame:
    kmer_columns = [col for col in df.columns if col.isupper() and len(col) == k]
    selected_columns = kmer_columns + ['resistant_phenotype', 'genbank_id']
    df = df[selected_columns]
    df.set_index('genbank_id', inplace=True)
    return df

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop('resistant_phenotype', axis=1)
    y = df['resistant_phenotype']

    # Calculate total size
    total_size = max(30, int(0.3 * len(df)))
    total_size = min(total_size, len(df))

    # Subset the data
    df = df.sample(n=total_size, random_state=42)

    # Update X and y
    X = df.drop('resistant_phenotype', axis=1)
    y = df['resistant_phenotype']

    # Calculate test size
    test_size = 0.3

    # Perform stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test

## Lazy Predict
def evaluate_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    # Replace None values with a default value
    models.fillna(0, inplace=True)

    # Print out the performance
    print(models)

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(models, annot=True, cmap='viridis')
    plt.title('Model Performance')
    plt.savefig('model_performance.png')

    return models'''
