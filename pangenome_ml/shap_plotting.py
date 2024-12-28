import os 
import pandas as pd 
import shap 
import matplotlib.pyplot as plt 

def plot_shap_values(classifier_model_name): 
    mean_shap_values.to_csv(f'{output_dir}/mean_shap_values.csv') 

    plt.figure(figsize=(10 ,6)) 
     
    mean_shap_values.head(20).plot(kind='barh') 
     
    plt.title('Top 20 Features by Mean SHAP Values') 
     
    plt.xlabel('Mean SHAP Value') 
     
    plt.ylabel('Features') 
     
    plt.savefig(f'{output_dir}/mean_shap_plot.png') 
