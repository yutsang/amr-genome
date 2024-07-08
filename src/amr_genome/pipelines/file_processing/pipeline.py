# src/my_project/pipelines/data_pipeline.py

from kedro.pipeline import Pipeline, node

#from .nodes import load_and_merge, process_data, stratified_sample
#from .nodes import get_and_set_link
from .nodes import kmer_processing, fna_line, create_annotation #process_files, 
from .nodes import stratified_split, merge_kmer_files, train
#from .nodes import merge_dfs, prepare_data, split_data, evaluate_models

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=fna_line,
                inputs="params:path_raw",
                outputs=None,
                name="fna_line",
            ),
            node(
                func=kmer_processing,
                inputs=["params:k", "params:rpath_raw", "params:path_kmer"],
                outputs=None,
                name="kmer_processing",
            ),
            node(
                func=create_annotation,
                inputs=["stratified_sample", "params:path_kmer"],
                outputs="annotation",
                name="create_annotation",
            ),
            node(
                func=stratified_split,
                inputs=["annotation", "params:train_ratio", "params:val_ratio", "params:test_ratio"],
                outputs=["train_data", "val_data", "test_data"],
                name="stratified_split",
            ),
            node(
                func=merge_kmer_files,
                inputs=["params:path_kmer", "params:path_merge_kmer"],
                outputs="merge_kmer_df",
                name="merge_kmer_files_node",
            ),
            node(
                func=train,
                inputs=["merge_kmer_df", "annotation"],
                outputs="model_results",
                name="train_node",
            ),
            #'''node(
            #    func=process_files,
            #    inputs=["params:k", "params:max_workers"],
            #    outputs="genomes_kmers",
            #    name="process_files",
            #),
            #node(
            #    func=merge_dfs,
            #    inputs=["merge_ftp_df", "genomes_kmers"],
            #    outputs="genomes_kmers_amr",
            #    name="merge_dfs",
            #),
            
            #node(
            #    func=prepare_data,
            #    inputs=["genomes_kmers_amr", "params:k"],
            #    outputs="prepared_data",
            #    name="prepare_data_node",
            #),
            #node(
            #    func=split_data,
            #    inputs=["prepared_data"],
            #    outputs=["X_s_train", "X_s_test", "y_s_train", "y_s_test"],
            #    name="split_data_node",
            #),
            #node(
            #    func=evaluate_models,
            #    inputs=["X_s_train", "X_s_test", "y_s_train", "y_s_test"],
            #    outputs="model_performance",
            #    name="evaluate_models_node",
            #),'''

        ]
    )