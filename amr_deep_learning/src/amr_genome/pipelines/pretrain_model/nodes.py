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
import itertools
from kedro.pipeline import node
from sklearn.model_selection import train_test_split
from typing import Tuple
from lazypredict.Supervised import LazyClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import glob

pd.options.mode.chained_assignment = None

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

    return models
