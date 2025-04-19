"""
Module for training and evaluating CATHODE models.
"""
import os
import datetime
import numpy as np
import pandas as pd
import pickle
import torch
import sklearn
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KernelDensity

from config import *
from utils.Helper_Functions import LogitScaler
from utils.Normalizing_Flow import ConditionalNormalizingFlow
from ensemble_trainer import train_and_compare_ensembles


def prepare_datasets(signal_data, background_data):
    """Convert dictionaries into pandas DataFrames and split into datasets."""
    from utils.Naming_Conventions import Simplified_Naming_Convention
    
    # Create datasets using the simplified naming convention function
    datasets = Simplified_Naming_Convention(signal_data, background_data)
    
    # Unpack the results and simplify to match our reduced model
    dataset_sig, dataset_bg = datasets[0], datasets[1]
    column_labels = datasets[5]  # Assuming this is the column labels index
    
    return {
        'dataset_sig': dataset_sig,
        'dataset_bg': dataset_bg,
        'column_labels': column_labels
    }


def make_train_test_val(sig, bkg1, column_labels):
    """Create training, validation, and test sets."""
    from utils.Data_Splitting import make_train_test_val_simplified
    
    train, val, test, feature_list, \
    outerdata_train, outerdata_val, innerdata_train, innerdata_val, innerdata_test, \
    Set_aside_background = make_train_test_val_simplified(
        sig, bkg1, 
        m_tt_min=LOWER_MASS_BOUND,
        m_tt_max=UPPER_MASS_BOUND,
        sig_injection=SIGNAL_INJECTION_RATIO, 
        testing_set_size=TEST_SET_RATIO,
        column_labels=column_labels,
        random_seed=DATA_SEED
    )
    
    return {
        'train': train,
        'val': val, 
        'test': test,
        'feature_list': feature_list,
        'outerdata_train': outerdata_train, 
        'outerdata_val': outerdata_val, 
        'innerdata_train': innerdata_train, 
        'innerdata_val': innerdata_val, 
        'innerdata_test': innerdata_test,
        'Set_aside_background': Set_aside_background
    }


def prepare_flow_model(outerdata_train, outerdata_val):
    """Set up and train or load a normalizing flow model."""
    outer_scaler = make_pipeline(LogitScaler(), StandardScaler())
    
    m_train = outerdata_train[:, 0:1]
    X_train = outer_scaler.fit_transform(outerdata_train[:, 1:-1])
    
    m_val = outerdata_val[:, 0:1]
    X_val = outer_scaler.transform(outerdata_val[:, 1:-1])
    
    inputs = -1  # Use all inputs
    
    # Create directory for saving models
    now = datetime.datetime.now()
    time_str = now.strftime("%H-%M-%S")
    
    # Set up file paths
    model_name = f'DE_{SIGNAL_DIRECTORY}_{LEARNING_RATE_PRINT}_{time_str}'
    
    local_file_path_to_save = os.path.join(CATHODE_BASE_DIRECTORY, PICKLE_FILE_BASE_DIRECTORY)
    flow_savedir = os.path.join(CATHODE_BASE_DIRECTORY, PICKLE_FILE_BASE_DIRECTORY, "cathode_models")
    
    # Create directories if they don't exist
    os.makedirs(local_file_path_to_save, exist_ok=True)
    os.makedirs(flow_savedir, exist_ok=True)
    
    if not os.path.exists(os.path.join(flow_savedir, "DE_models")) and not LOAD_TRAINED_MODEL:
        # Create a new flow model
        print("Starting flow model training...")
        
        flow_model = ConditionalNormalizingFlow(
            save_path=flow_savedir,
            num_inputs=outerdata_train[:, 1:inputs].shape[1],
            early_stopping=EARLY_STOPPING,
            verbose=True,
            tail_bound=TAIL_BOUND_VALUE,
            num_bins=NUMBER_OF_BINS,
            num_hidden=NUM_HIDDEN,
            num_blocks=NUM_BLOCKS,
            num_layers=NUM_LAYERS,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            run_name=model_name,
            lr=LEARNING_RATE,
            num_cond_inputs=NUM_COND_INPUTS,
            num_cycles=NUM_CYCLES,
            patience=PATIENCE
        )
        
        # Train the model
        flow_model.fit_noPenalty(X_train, m_train, X_val, m_val)
    else:
        print(f"Loading existing model from {flow_savedir}")
    
    # Load the model
    flow_model = ConditionalNormalizingFlow(
        save_path=flow_savedir,
        num_inputs=outerdata_train[:, 1:inputs].shape[1],
        early_stopping=False,
        verbose=True,
        tail_bound=TAIL_BOUND_VALUE,
        num_bins=NUMBER_OF_BINS,
        num_hidden=NUM_HIDDEN,
        num_blocks=NUM_BLOCKS,
        num_layers=NUM_LAYERS,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS, 
        run_name=model_name,
        lr=LEARNING_RATE,
        num_cond_inputs=NUM_COND_INPUTS,
        load=True,
        num_cycles=NUM_CYCLES
    )
    
    return flow_model, outer_scaler


def generate_samples(flow_model, outer_scaler, innerdata_train):
    """Generate samples using the trained flow model."""
    m_scaler = LogitScaler(epsilon=1e-8)
    m_train = m_scaler.fit_transform(innerdata_train[:, 0:1])
    kde_model = KernelDensity(bandwidth=0.01, kernel='gaussian')
    kde_model.fit(m_train)
    
    # Determine number of samples to generate
    training_inner_data = innerdata_train.shape[0]
    training_data_size = int(training_inner_data * 0.85)
    
    # Generate samples
    m_samples = kde_model.sample(4 * training_data_size).astype(np.float32)
    m_samples = m_scaler.inverse_transform(m_samples)
    X_samples = flow_model.sample(n_samples=len(m_samples), m=m_samples)
    X_samples_2 = X_samples.copy()
    
    # Invert scaling
    X_samples = outer_scaler.inverse_transform(X_samples)
    samples = np.hstack([m_samples, X_samples, np.zeros((m_samples.shape[0], 1))])
    
    return samples, X_samples_2


def evaluate_cathode_background_quality(innerdata_test, samples, reference_data=None, column_labels=None):
    """
    Evaluate the quality of CATHODE-generated background samples by comparing them to real data.
    
    Parameters:
    -----------
    innerdata_test : numpy.ndarray
        Test data to compare samples against
    samples : numpy.ndarray
        CATHODE-generated samples to evaluate
    reference_data : numpy.ndarray, optional
        Additional reference data for comparison
    column_labels : list, optional
        Labels for the columns in the data
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics and paths to generated plots
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    print("Evaluating CATHODE background quality...")
    
    # Set up the output directory for plots
    plot_dir = os.path.join(PLOT_DIRECTORY, SIGNAL_DIRECTORY, "background_quality")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Generate a timestamp for the output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Compare samples with test data
    print("Comparing features between generated samples and test data...")
    
    # Calculate quality metrics
    print("Calculating quality metrics...")
    quality_metrics = calculate_quality_metrics(innerdata_test, samples)
    
    comparison_results = {
        'metrics': quality_metrics,
        'plot_dir': plot_dir,
        'timestamp': timestamp
    }
    
    print("Background quality evaluation complete!")
    return comparison_results


def calculate_quality_metrics(data, samples):
    """
    Calculate quality metrics for CATHODE-generated samples.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Reference data
    samples : numpy.ndarray
        Generated samples
        
    Returns:
    --------
    dict
        Dictionary containing quality metrics
    """
    import numpy as np
    from scipy.stats import ks_2samp, wasserstein_distance
    
    metrics = {}
    
    # Calculate mean and standard deviation differences
    data_means = np.nanmean(data, axis=0)
    data_stds = np.nanstd(data, axis=0)
    samples_means = np.nanmean(samples, axis=0)
    samples_stds = np.nanstd(samples, axis=0)
    
    metrics['mean_diff'] = np.abs(samples_means - data_means)
    metrics['mean_diff_normalized'] = np.abs((samples_means - data_means) / data_stds)
    metrics['std_ratio'] = samples_stds / data_stds
    
    # Calculate KS statistics for each feature
    ks_stats = []
    p_values = []
    wasserstein_distances = []
    
    for i in range(data.shape[1]):
        data_col = data[:, i][~np.isnan(data[:, i])]
        samples_col = samples[:, i][~np.isnan(samples[:, i])]
        
        # If data is empty or constant, skip
        if len(data_col) < 2 or len(samples_col) < 2:
            ks_stats.append(np.nan)
            p_values.append(np.nan)
            wasserstein_distances.append(np.nan)
            continue
            
        # Kolmogorov-Smirnov test
        ks_stat, p_value = ks_2samp(data_col, samples_col)
        ks_stats.append(ks_stat)
        p_values.append(p_value)
        
        # Wasserstein distance (Earth Mover's Distance)
        try:
            w_dist = wasserstein_distance(data_col, samples_col)
            wasserstein_distances.append(w_dist)
        except Exception:
            wasserstein_distances.append(np.nan)
    
    metrics['ks_stats'] = np.array(ks_stats)
    metrics['p_values'] = np.array(p_values)
    metrics['wasserstein_distances'] = np.array(wasserstein_distances)
    
    # Calculate correlation matrix differences
    data_corr = np.corrcoef(data, rowvar=False)
    samples_corr = np.corrcoef(samples, rowvar=False)
    corr_diff = samples_corr - data_corr
    
    metrics['corr_diff'] = corr_diff
    metrics['corr_diff_mean'] = np.nanmean(np.abs(corr_diff))
    metrics['corr_diff_max'] = np.nanmax(np.abs(corr_diff))
    
    # Overall quality score (lower is better)
    # Weighted combination of normalized mean differences, KS stats, and correlation differences
    mean_diff_score = np.nanmean(metrics['mean_diff_normalized'])
    ks_score = np.nanmean(metrics['ks_stats'])
    corr_score = metrics['corr_diff_mean']
    
    metrics['overall_score'] = 0.4 * mean_diff_score + 0.4 * ks_score + 0.2 * corr_score
    
    return metrics