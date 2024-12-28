# AMR_ML_KP11 v4.2.2 (on 22 Dec 2024)

## Background

AMR_ML_KP11 is a machine learning project designed to analyze antimicrobial resistance (AMR) data using various classification models. The project utilizes data from Excel files containing metadata and pangenome information to train and evaluate models such as Decision Trees, Random Forests, Gradient Boosting, Support Vector Machines (SVM), and XGBoost. The results include accuracy metrics and SHAP value analyses for feature importance interpretation.

## Features

- Data processing from Excel files with multiple sheets.
- Support for various classification algorithms.
- SHAP value analysis to interpret model predictions.
- Output of model performance metrics and SHAP plots.

## Requirements

To run this project, you need the following Python packages:

- pandas
- scikit-learn
- xgboost
- shap
- matplotlib
- openpyxl

### Environment Setup

To simplify the setup process, an `environment.yml` file is provided. You can create a conda environment with all the required dependencies by running:

```conda env create -f environment.yml```

Activate the environment using:

```conda activate AMR_ML_KP11```

Alternatively, you can install the required packages using pip:

```pip install pandas scikit-learn xgboost shap matplotlib openpyxl```


## Directory Structure

AMR_ML_KP11/   
├── input/ # Directory for input Excel files  
├── output/ # Directory for output results (CSV, plots)  
├── data_processing.py # Module for processing input data  
├── model_training_and_shap.py # Module for training models and SHAP analysis  
└── main.py # Main entry point to run the application  
├── environment.yml # Conda environment configuration file  

## How to Run the Model

1. **Prepare Your Data**: Place your Excel file in the `input/` directory. Ensure that the file contains exactly two sheets: one for metadata and one for pangenome data.

2. **Run the Application**: Execute the main script to process the data, train the models, and generate outputs.

```python main.py```


3. **View Results**: After running the script, check the `output/` directory for:
   - CSV files containing model performance metrics.
   - Plots showing feature importance based on SHAP values.

## Example Usage

There is a sample .xlsx in the root directory with minimum content for demonstration purposes:

Place your Excel file in the input directory  
```mv data.xlsx input/```  
Run the model training and evaluation  
```python main.py```  

## Most Recent Update (v4.2.2)
- Automatic Detection of Multiple XLSX Files: Improved error handling to manage scenarios where more than one XLSX file is detected, ensuring that only the first file is processed.
- Utilization of OpenPyXL: All operations are now performed using the OpenPyXL library for better compatibility and performance with XLSX files.
- Automatic Column Name Adjustment: The script now automatically modifies column names to create features, enhancing usability.
- Worksheet Detection: The system now automatically identifies worksheets, categorizing them into "pangenome" (for those with more data) and "metadata" (for those with less data).
- Transposing Pangenome Worksheet: The pangenome worksheet is transposed to optimize processing speed and efficiency.
- Feature Conversion to int8: Features are converted to int8 data type to reduce memory usage in SVM operations, addressing memory issues effectively.
- Automatic Merging of Data: Added functionality for automatic merging of relevant data, streamlining the workflow and enhancing data management capabilities.


## Notes

- Ensure that your Excel file is formatted correctly with appropriate headers.
- The target columns are defined within the code; modify them as needed based on your dataset.
- If you encounter any issues, check that all required libraries are installed and that your input data meets the expected format.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project utilizes various machine learning techniques and libraries. Special thanks to the contributors of these libraries.
