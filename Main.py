"""
Main script for CATHODE analysis.
"""
import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.multiprocessing import set_start_method

# Import configuration
from config import *

# Import utilities
from data_loader import load_or_create_data
from phase_space_processor import prepare_datasets
from model_trainer import make_train_test_val, prepare_flow_model
from model_trainer import generate_samples, evaluate_cathode_background_quality
from model_trainer import train_and_compare_ensembles


def main():
    """Main execution function for CATHODE analysis."""
    print("Starting CATHODE analysis...")
    
    # Set multiprocessing start method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  # Already set
    
    # Step 1: Load or create data
    print("\nStep 1: Loading or creating data...")
    signal_data, background_data = load_or_create_data()
    
    # Step 2: Prepare datasets
    print("\nStep 2: Preparing datasets...")
    datasets = prepare_datasets(signal_data, background_data)
    
    sig = datasets['dataset_sig']
    bkg1 = datasets['dataset_bg']
    column_labels = datasets['column_labels']
    
    # Step 3: Create training and test sets
    print("\nStep 3: Creating training and test sets...")
    data_splits = make_train_test_val(sig, bkg1, column_labels)
    
    # Get the appropriate datasets
    innerdata_train = data_splits['innerdata_train']
    innerdata_val = data_splits['innerdata_val']
    innerdata_test = data_splits['innerdata_test']
    outerdata_train = data_splits['outerdata_train']
    outerdata_val = data_splits['outerdata_val']
    
    # Step 4: Train or load flow model
    print("\nStep 4: Preparing flow model...")
    flow_model, outer_scaler = prepare_flow_model(outerdata_train, outerdata_val)
    
    # Step 5: Generate samples
    print("\nStep 5: Generating samples...")
    samples, samples_scaled = generate_samples(flow_model, outer_scaler, innerdata_train)
    
    # Step 6: Evaluate background quality
    print("\nStep 6: Evaluating background quality...")
    bg_eval_results = evaluate_cathode_background_quality(
        innerdata_test, 
        samples, 
        innerdata_test, 
        column_labels
    )
    
    # Step 7: Train and evaluate classifiers
    if USE_BDT:
        print("\nStep 7: Training and evaluating classifiers...")
        ensemble_results, comparison_dir = train_and_compare_ensembles(
            sig, bkg1,
            column_labels, 
            samples,
            n_dim=NUMBER_OF_DIMENSIONS,
            signal_injection_ratio=SIGNAL_INJECTION_RATIO,
            bdt_models_per_set=5,
            test_fraction=0.25,
            use_bdt=USE_BDT,
            m_tt_min=LOWER_MASS_BOUND,
            m_tt_max=UPPER_MASS_BOUND,
            plot_directory=PLOT_DIRECTORY,
            random_seed=DATA_SEED,
            Signal_Directory=SIGNAL_DIRECTORY,
            mode="train"
        )
        
        print(f"\nResults saved to: {comparison_dir}")
    
    print("\nAnalysis complete!")
    return datasets


if __name__ == "__main__":
    results = main()