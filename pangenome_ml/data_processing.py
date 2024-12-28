import os
import pandas as pd

def process_excel_file():
    input_dir = './input'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    root_dir = '.'
    
    files = [f for f in os.listdir(root_dir) if f.endswith('.xlsx')]

    # Check if no Excel files are found
    if len(files) == 0:
        raise FileNotFoundError("No Excel files found in the specified directory.")

    # Check if more than one Excel file is found
    if len(files) > 1:
        raise ValueError("Multiple Excel files found. Please ensure there is only one .xlsx file in the directory.")

    # Load the first (and only) Excel file
    excel_file_path = os.path.join(root_dir, files[0])

    # Check for the number of sheets in the Excel file
    try:
        xls = pd.ExcelFile(excel_file_path, engine='openpyxl')
        sheet_names = xls.sheet_names

        # Check if there are exactly two sheets
        if len(sheet_names) != 2:
            raise ValueError("The Excel file must contain exactly two sheets.")

        # Read both sheets into DataFrames
        metadata_df = pd.read_excel(xls, sheet_name=sheet_names[0])
        pangenome_df = pd.read_excel(xls, sheet_name=sheet_names[1])

        # Clean both DataFrames by checking and removing row number columns
        metadata_df = check_and_clean_first_column(metadata_df)
        pangenome_df = check_and_clean_first_column(pangenome_df)

        # Rename binary columns in metadata
        target_cols = rename_binary_columns(metadata_df)

        # Get keys for merging
        metadata_key = get_first_valid_column(metadata_df)
        
        # Transpose pangenome and set 'Gene' as index (first column)
        pangenome_key = pangenome_df.columns[0]
        pangenome_transposed = pangenome_df.set_index(pangenome_key).transpose()

        # Check for completeness of IDs before merging
        check_id_completeness(metadata_df, pangenome_transposed, metadata_key)

        # Proceed with merging using left_on for metadata's key and right_index for pangenome's index after transpose
        merged_df = pd.merge(metadata_df, pangenome_transposed.reset_index(), left_on=metadata_key, right_on='index', how='inner').drop(columns=['index'])
        
        save_to_csv(input_dir, metadata_df, pangenome_transposed, merged_df)

        return metadata_df, pangenome_transposed, merged_df, target_cols

    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the Excel file: {e}")

def check_and_clean_first_column(df):
    """Check if the first column is a row number and drop it if so."""
    first_col = df.iloc[:, 0]
    if pd.api.types.is_integer_dtype(first_col) and first_col.is_monotonic_increasing and (first_col.iloc[0] == 0):
        df = df.iloc[:, 1:]  # Drop the first column
    return df

def get_first_valid_column(df):
    """Return the name of the first valid column that is not a row number."""
    first_col = df.iloc[:, 0]
    if pd.api.types.is_integer_dtype(first_col) and first_col.is_monotonic_increasing and (first_col.iloc[0] == 0):
        return df.columns[1]
    return df.columns[0]

def rename_binary_columns(metadata):
    """Rename binary columns based on their unique values."""
    binary_columns = []
    for col in metadata.columns[1:]:  # Skip the first column which is ID
        unique_values = metadata[col].unique()
        if len(unique_values) == 2:  # Check for binary values
            lower_values = [str(value).lower() for value in unique_values]
            shorter_value = min(lower_values, key=len)
            metadata.rename(columns={col: shorter_value}, inplace=True)
            binary_columns.append(shorter_value)
    
    return binary_columns

def check_id_completeness(metadata_df, pangenome_transposed, metadata_key):
    """Check for completeness of IDs before merging."""
    metadata_ids = set(metadata_df[metadata_key])
    
    # Use index of transposed pangenome for matching
    pangenome_ids = set(pangenome_transposed.index)

    missing_in_metadata = pangenome_ids - metadata_ids
    missing_in_pangenome = metadata_ids - pangenome_ids

    # Output any mismatches
    if missing_in_metadata or missing_in_pangenome:
        print("Mismatches found:")
        if missing_in_metadata:
            print(f"IDs in pangenome not found in metadata: {missing_in_metadata}")
        if missing_in_pangenome:
            print(f"IDs in metadata not found in pangenome: {missing_in_pangenome}")
    
def save_to_csv(input_dir, metadata_df, pangenome_transposed, merged_df):
    """Save DataFrames to CSV."""
    os.makedirs(input_dir, exist_ok=True)
    
    metadata_df.to_csv(os.path.join(input_dir, 'metadata.csv'))
    pangenome_transposed.to_csv(os.path.join(input_dir, 'pangenome.csv'))
    merged_df.to_csv(os.path.join(input_dir, 'merged.csv'))
