# FNA Sequence Processor

This repository contains a Python script for processing `.fna` files containing nucleotide sequences. The script utilizes a pre-trained model from the Hugging Face Transformers library to classify sequences and extract features. The results are saved in a CSV format for further analysis.

## Features

- Process `.fna` files to classify nucleotide sequences.
- Option to separate sequences by contig or treat the entire file as a single input.
- Extract hidden state features from the model for each token in the sequence.
- Save results, including predictions and features, in a CSV file.

## Requirements

- Conda (recommended for managing dependencies)
- Python 3.6 or higher
- PyTorch
- Transformers
- Pandas

## Conda Environment Setup

To create a new Conda environment for this project, follow these steps:

1. **Create an environment configuration file**: The following YAML configuration is provided to set up the environment:

```
name: fna_processor_env
channels:
- defaults
- conda-forge
dependencies:
- python=3.8
- pytorch
- transformers
- pandas
```

2. **Save the above configuration** to a file named `environment.yml`.

3. **Create the Conda environment** by running the following command in your terminal:
```
conda env create -f environment.yml
```

4. **Activate the environment** with:
```
conda activate fna_processor_env
```


## Usage

### Command-Line Arguments

The script accepts the following command-line arguments:

- `-m`, `--model_directory`: Path to the directory containing the pre-trained model files (required).
- `-f`, `--fna_file_path`: Path to the input `.fna` file containing nucleotide sequences (required).
- `-o`, `--output_directory`: Directory where the output CSV file will be saved (required).
- `-s`, `--separate_by_contig`: Optional flag to process each contig (sequence) separately. If not provided, all sequences will be combined into one.

### Running the Script

1. **Clone the repository** or download the script file.
2. **Prepare your model**: Ensure you have a pre-trained model saved in a directory.
3. **Prepare your `.fna` file**: Have your input file ready in FASTA format.

#### Example Commands

**Full FNA as One Single Input**
python check_bert.py -m ./model/ -f ./input/GCF_000781775.1.fna -o ./output/

### Output

The results will be saved in a CSV file named `<base_filename>_results.csv` in the specified output directory, where `<base_filename>` is derived from your input `.fna` filename without its extension. The CSV will contain:

- **filename**: Name of the input file.
- **input_sequence**: A snippet of the input sequence (first 50 characters).
- **predicted_class**: The predicted class for each sequence.
- **features**: The feature vectors for each token in the sequence.

## Example Output

If your input filename is `GCF_000781775.1.fna`, you will find an output file named `GCF_000781775.1_results.csv` in your specified output directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This script utilizes models and tools from Hugging Face's Transformers library and PyTorch.

