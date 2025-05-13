import logging
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("MLLogger")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def has_nvidia_gpu():
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def display_feature_importances(model, feature_names, model_name="Model", top_n=15):
    if not hasattr(model, 'feature_importances_'):
        return

    importances = model.feature_importances_
    
    if len(feature_names) != len(importances):
        logger.error(f"Feature name/importance length mismatch for {model_name}.")
        feature_names = [f"feature_{i}" for i in range(len(importances))] 

    feature_importance_map = dict(zip(feature_names, importances))
    sorted_features = sorted(feature_importance_map.items(), key=lambda item: item[1], reverse=True)

    # Logging still respects top_n for console output
    logger.info(f"--- Top {min(top_n, len(sorted_features))} Feature Importances for {model_name} (Logged) ---")
    for i in range(min(top_n, len(sorted_features))):
        feature_name, importance_score = sorted_features[i]
        logger.info(f"{i+1}. Feature: {feature_name}, Importance: {importance_score:.4f}")
    logger.info("----------------------------------------------------")

    if not sorted_features:
        return

    plot_feature_names = [item[0] for item in sorted_features]
    plot_importances = [item[1] for item in sorted_features]
    num_features_to_plot = len(plot_feature_names)

    plt.figure(figsize=(12, max(8, num_features_to_plot * 0.3))) 
    plt.title(f'Feature Importances - {model_name} ({num_features_to_plot} features)')
    
    plt.barh(np.arange(num_features_to_plot), plot_importances[::-1], align='center')
    plt.yticks(np.arange(num_features_to_plot), plot_feature_names[::-1])
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout() 
    
    plot_filename = f"feature_importance_{model_name.replace(' ', '_')}_all.png"
    try:
        plt.savefig("../Plots/" + plot_filename)
        logger.info(f"Feature importance plot for {model_name} saved to {plot_filename}")
    except Exception as e:
        logger.error(f"Could not save plot for {model_name}: {e}")
    plt.close()