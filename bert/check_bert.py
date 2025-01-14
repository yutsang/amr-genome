from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import os
import argparse

# Function to read sequences from a .fna file, either separated by contig or as a single input
def read_fna_file(file_path, separate_by_contig):
    sequences = []
    with open(file_path, 'r') as file:
        seq = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):  # Header line
                if seq:  # If there is a sequence collected, add it to the list
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line  # Collect sequence lines
        if seq:  # Add the last sequence if exists
            sequences.append(seq)

    if not separate_by_contig:  # Combine all sequences into one if not separating by contig
        sequences = ["".join(sequences)]
    
    return sequences

# Main function to process the .fna file and save predictions and features to CSV
def main(args):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_directory)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_directory, output_hidden_states=True)

    # Read sequences from the .fna file
    sequences = read_fna_file(args.fna_file_path, args.separate_by_contig)

    # Prepare a list to hold results for DataFrame
    results = []

    # Variable to track final prediction
    final_prediction = 0

    # Get predictions and hidden states for each sequence
    for text in sequences:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(dim=-1).item()

            # Check if any predicted class is 1 for final prediction
            if predicted_class == 1:
                final_prediction = 1

            # Extract hidden states
            hidden_states = outputs.hidden_states  # This will be a tuple of hidden states from each layer

        # Process hidden states for feature analysis (example using the last layer)
        last_layer_hidden_states = hidden_states[-1]  # Get hidden states from the last layer

        # Prepare feature vector for each token in the input sequence
        feature_vectors = []
        for i, token in enumerate(tokenizer.tokenize(text)):
            feature_vector = last_layer_hidden_states[0][i].tolist()  # Convert tensor to list for display
            feature_vectors.append(feature_vector)

        # Append results to the list with filename, input sequence snippet, predicted class, and features
        result_entry = {
            'filename': os.path.basename(args.fna_file_path),  # Extracting filename from path
            'input_sequence': text[:50],  # Show only first 50 characters of input sequence
            'predicted_class': predicted_class,
            'features': feature_vectors
        }

        results.append(result_entry)

    # Create a DataFrame from results
    df = pd.DataFrame(results)

    # Construct output CSV filename based on input filename
    base_filename = os.path.splitext(os.path.basename(args.fna_file_path))[0]  # Get base name without extension
    output_csv_path = os.path.join(args.output_directory, f"{base_filename}_results.csv")

    # Save DataFrame to CSV file
    df.to_csv(output_csv_path, index=False)

    # Print final prediction result (0 or 1)
    print(f"Final Prediction: {final_prediction}")
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .fna file and save predictions and features to CSV.")
    
    parser.add_argument("-m", "--model_directory", type=str, required=True, help="Path to the model directory.")
    
    parser.add_argument("-f", "--fna_file_path", type=str, required=True, help="Path to the .fna file.")
    
    parser.add_argument("-o", "--output_directory", type=str, required=True, help="Directory to save the output CSV.")
    
    #parser.add_argument("-s", "--separate_by_contig", action="store_true", help="Flag to separate sequences by contig.")

    args = parser.parse_args()
    main(args)
