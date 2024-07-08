# src/my_project/pipelines/data_pipeline.py

from kedro.pipeline import Pipeline, node

#from .nodes import load_and_merge, process_data, stratified_sample
#from .nodes import get_and_set_link
#from .nodes import merge_dfs, prepare_data, download_and_unzip, process_files
from .nodes import split_data, evaluate_models

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            
            
            node(
                func=split_data,
                inputs=["prepared_data"],
                outputs=["X_s_train", "X_s_test", "y_s_train", "y_s_test"],
                name="split_data_node",
            ),
            node(
                func=evaluate_models,
                inputs=["X_s_train", "X_s_test", "y_s_train", "y_s_test"],
                outputs="model_performance",
                name="evaluate_models_node",
            ),

        ]
    )