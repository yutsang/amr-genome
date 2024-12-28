# amr genome

## Overview

The Kedro project is about amr genome prediction managed with Kedro-Viz and PySpark setup, which was generated using `kedro 0.19.5`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the repository:

* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

For safety, I also use conda to generate the environment.yml to restore the environment.

## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. Install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to run your Kedro pipeline

After changing the parameters, you can run your Kedro project with:

```
kedro run
```

However, for better pipeline management, best practice is to:
```
kedro run --pipeline=data_processing
kedro run --pipeline=file_processing
```
## To visualize the pipeline

To better taking a visual representation of the pipeline data processing:
```
kedro viz run
```

## How to adjust the parameters

All configuration documents are located inside the "/conf/" folder and mostly I have used are inside "/conf/base/".

"catalog.yml" contains the file definition including filename and the path while "parameters.yml" contains some key values are being used in the kedro pipeline. And, "catalog_*.yml" and "parameters_*.yml" mean the local catalog and parameters files specifically for the data pipeline "*".

Inside the "/conf/base/parameters.yml", the best configurations are:
* k: 11 (k used in kmer) 
* max_workers: 18 (cpu core used during multiprocessing)
* test_size: 4500 (cap at max(row count))
* antibiotic: "tetracycline" (for filtering)
* df_antibiotic_metadata_path: "./data/01_raw/patric2021_cleaned_1b.xlsx" (input path, formatted input)
* antibiotic_class_json_path: "./conf/base/antibiotic_classes.json" (GPT-created antibiotic classification)
* path_raw: "./data/02_intermediate/fna" (fna files would be downloaded in this relative path)
* rpath_raw: "./data/02_intermediate/fna_line" (line the fna files and would contain in this folder)
* path_kmer: "./data/02_intermediate/kmer" (kmer counting by jellyfish would be generated in this folder)
* path_merge_kmer: "./data/02_intermediate/" (folder contains intermediate processed files)
* train_ratio: 0.64, val_ratio: 0.16, test_ratio: 0.2 (train_val_test ratios)
