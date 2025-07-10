import os
import pandas as pd
import requests
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def download_file(locus):
    base_url = "https://bigsdb.pasteur.fr/cgi-bin/bigsdb/bigsdb.pl?db=pubmlst_elizabethkingia_seqdef&page=downloadAlleles&locus="
    url = base_url + locus
    filename = f"download/{locus}.fas"
    
    if os.path.exists(filename):
        return f"File already exists, skipping: {filename}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return f"Downloaded: {filename}"
    else:
        return f"Failed to download: {locus}"

def main():
    # Find the Excel file in the current directory
    excel_file = None
    for file in os.listdir('.'):
        if file.endswith('.xlsx'):
            excel_file = file
            break
    
    if not excel_file:
        print("No Excel file found in the current directory.")
        return
    
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    # Create a directory to store the downloaded files if it doesn't exist
    os.makedirs('download', exist_ok=True)
    
    # Use multiprocessing to download files with a progress bar
    loci = df['locus'].tolist()
    with Pool(cpu_count()) as pool:
        # Wrap the imap_unordered call with tqdm for progress bar
        for _ in tqdm(pool.imap_unordered(download_file, loci), total=len(loci)):
            pass

if __name__ == "__main__":
    main()
