# src/my_project/pipelines/data_pipeline.py

from kedro.pipeline import Pipeline, node

#from .nodes import load_and_merge, process_data, stratified_sample
from .nodes import get_and_set_link, load_process_sample, download_and_unzip #, process_files
#from .nodes import merge_dfs, prepare_data, split_data, evaluate_models

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_process_sample,
                inputs=["params:df_antibiotic_metadata_path", 
                        "params:antibiotic_class_json_path", 
                        "params:test_size",
                        "params:antibiotic"],
                outputs="stratified_sample",
                name="load_process_sample",
            ),
            node(
                func=get_and_set_link,
                inputs=["stratified_sample", "params:max_workers"], 
                outputs="merge_ftp_df",
                name="get_and_set_link",
            ),
            node(
                func=download_and_unzip,
                inputs=["merge_ftp_df", "params:max_workers"],
                outputs=None,
                name="download_and_unzip",
            ),

        ]
    )