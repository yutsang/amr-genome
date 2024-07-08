# Import necessary libraries
import os
import pandas as pd
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import umap
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# Function to parse a single .fna file
def parse_fna_file(filepath):
    sequences = []
    filenames = []
    for record in SeqIO.parse(filepath, "fasta"):
        sequences.append(str(record.seq))
        filenames.append(os.path.basename(filepath))
    return sequences, filenames

# Function to parse all .fna files using multiprocessing
def parse_fna_files(directory):
    filepaths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".fna")]
    with Pool() as pool:
        results = list(tqdm(pool.imap(parse_fna_file, filepaths), total=len(filepaths), desc="Parsing .fna files"))
    sequences = [seq for result in results for seq in result[0]]
    filenames = [fname for result in results for fname in result[1]]
    return sequences, filenames

# Function to get k-mers from a sequence
def get_kmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# Function to extract features from sequences using multiprocessing
def extract_features(sequences, k=6):
    with Pool() as pool:
        kmers = list(tqdm(pool.imap(lambda seq: ' '.join(get_kmers(seq, k)), sequences), total=len(sequences), desc="Extracting k-mers"))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(kmers)
    return X, vectorizer

def main():
    # Directory containing your .fna files
    fna_directory = './fna/'

    # Parse .fna files
    sequences, filenames = parse_fna_files(fna_directory)

    # Extract features
    X, vectorizer = extract_features(sequences)

    # Load labels from a CSV file
    labels_df = pd.read_csv('label.csv')
    labels_df['resistant_phenotype'] = labels_df['resistant_phenotype'].map({'resistant': 1, 'susceptible': 0})

    # Map labels to sequences based on filenames
    labels = []
    for filename in tqdm(filenames, desc="Mapping labels"):
        genbank_id = '.'.join(filename.split('.')[:2])  # Correctly handle filenames with multiple periods
        label_row = labels_df.loc[labels_df['genbank_id'] == genbank_id, 'resistant_phenotype']
        if not label_row.empty:
            label = label_row.values[0]
            labels.append(label)
        else:
            print(f"Warning: GenBank ID {genbank_id} not found in labels file. Skipping this file.")
            # Optionally, you can append a default label or handle it differently
            # labels.append(default_label)

    # Convert labels to a numpy array and ensure they are numerical
    y = pd.Series(labels).astype(float).values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Perform UMAP for dimension reduction
    reducer = umap.UMAP(n_components=100, random_state=42)  # Adjust the number of components as needed
    X_train_umap = reducer.fit_transform(X_train_scaled)
    X_test_umap = reducer.transform(X_test_scaled)

    # Define the deep learning model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_umap.shape[1],)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model with a progress bar
    history = model.fit(X_train_umap, y_train, epochs=10, batch_size=32, validation_data=(X_test_umap, y_test), verbose=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_umap, y_test)
    print(f'Test Accuracy: {accuracy:.4f}')

    # Save the model
    model.save('amr_prediction_model.h5')

    # Load the model for future use
    # model = load_model('amr_prediction_model.h5')

    # Make predictions
    # predictions = model.predict(X_test_umap)

if __name__ == '__main__':
    main()
