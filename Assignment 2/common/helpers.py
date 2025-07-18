import logging
import sys
import subprocess
import numpy as np
import matplotlib
import os
# Disable maplotlib GUI, since this creates 'main thread not in main loop' error.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Prepare folder for plots
os.makedirs("../Plots/", exist_ok=True)

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

def print_pos_and_pred(x_test_predictions, x_test_positions, amount=20):
    toprint = x_test_positions.sort_values(by=["srch_id"])[:amount]
    toprint["predictions"] = x_test_predictions[:amount]
    print(toprint.sort_values(by=["srch_id", "position"]))

def create_book_feature(df):
    conditions = [
        df['booking_bool'] == 1,
        df['click_bool'] == 1
    ]

    choices = [5, 1]

    df['book_feature'] = np.select(conditions, choices, default=0)
    df.drop(columns=['booking_bool', 'click_bool'], inplace=True, errors='ignore')
    return df

def resampling_bias_mitigation(data):
    # Split the dataset
    chain_hotels = data[data['prop_brand_bool'] == 1]
    ind_hotels = data[data['prop_brand_bool'] == 0]

    # Calculate how many independent rows are needed to match target ratio
    n_needed = len(chain_hotels) - len(ind_hotels)

    # Bootstrap (resample with replacement)
    ind_hotels_bootstrapped = resample(
        ind_hotels,
        replace=True,
        n_samples=n_needed,
        random_state=42
    )

    # Concatenate original data with bootstrapped independents
    df_balanced = pd.concat([data, ind_hotels_bootstrapped])

    return df_balanced
