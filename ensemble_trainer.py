"""
Ensemble trainer module for CATHODE background estimation.
Contains functions for training and comparing different ensemble models.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle
import json
import glob

# Import needed for BDT models
from utils.boosted_decision_tree import HGBClassifier
from utils.ensembling_utils import EnsembleModel

# Check if TensorFlow is available
try:
    import tensorflow as tf
    import tensorflow.keras as keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - Neural Network models will be disabled")


def train_and_compare_ensembles(sig, bkg1, column_labels, samples, n_dim, signal_injection_ratio=0.01, n_ensemble_sets=3, bdt_models_per_set=5, test_fraction=0.25, use_bdt=True, m_tt_min=0.175, m_tt_max=0.3, plot_directory="./plots", random_seed=42, Signal_Directory='test', mode="train", model_name=None, models_base_path="./bdt_models", **kwargs):
    """
    Train and compare BDT ensembles using the CATHODE approach.
    Can either train new models or load existing ones.
    
    Parameters:
    -----------
    sig, bkg1 : pandas.DataFrame
        Signal and background dataframes
    column_labels : list
        List of column names for the dataframes
    samples : numpy.ndarray
        CATHODE-generated samples
    n_dim : int
        Number of dimensions in the data
    signal_injection_ratio : float
        Fraction of signal to inject into training data (default: 0.01)
    n_ensemble_sets : int
        Number of ensemble sets to train (default: 3)
    bdt_models_per_set : int
        Number of BDT models per ensemble set (default: 5)
    test_fraction : float
        Fraction of data to use for testing (default: 0.25)
    use_bdt : bool
        Whether to train BDT models (default: True)
    m_tt_min, m_tt_max : float
        Mass window for signal region (default: 0.175, 0.3)
    plot_directory : str
        Directory to save plots (default: "./plots")
    random_seed : int
        Random seed for reproducibility (default: 42)
    Signal_Directory : str
        Directory name for signal files (default: 'test')
    mode : str
        'train' to train new models, 'load' to load existing models (default: "train")
    model_name : str
        Name of the model directory to load (only used if mode='load')
    models_base_path : str
        Base path for model storage (default: "./bdt_models")
    **kwargs : dict
        Additional options
        
    Returns:
    --------
    results : dict
        Dictionary containing results
    comparison_dir : str
        Directory containing comparison plots
    """
    # Start timing
    start_time = time.time()
    time_str = time.strftime("%Y%m%d_%H%M%S")
    
    # Create dictionary to store results
    results = {
        'mode': mode,
        'signal_directory': Signal_Directory,
        'signal_injection_ratio': signal_injection_ratio,
        'timestamp': time_str
    }
    
    # Set default train/val split
    Train_val_split = kwargs.get('Train_val_split', 0.5)
    
    # Determine model directory
    if mode == "train":
        # Create a new directory for this run
        model_dir_name = f"{Signal_Directory}_inj{int(signal_injection_ratio*1000)}_{time_str}"
        model_base_dir = os.path.join(models_base_path, model_dir_name)
        os.makedirs(model_base_dir, exist_ok=True)
        print(f"Models will be saved to: {model_base_dir}")
        
        # Save metadata about this run
        metadata = {
            'mode': mode,
            'signal_directory': Signal_Directory,
            'signal_injection_ratio': signal_injection_ratio,
            'test_fraction': test_fraction,
            'n_ensemble_sets': n_ensemble_sets,
            'bdt_models_per_set': bdt_models_per_set,
            'n_dim': n_dim,
            'm_tt_min': m_tt_min,
            'm_tt_max': m_tt_max,
            'random_seed': random_seed,
            'timestamp': time_str,
        }
        
        with open(os.path.join(model_base_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    elif mode == "load":
        if model_name is None:
            # If model_name is not provided, try to find the most recent model for this signal directory
            model_pattern = os.path.join(models_base_path, f"{Signal_Directory}_inj*")
            matching_dirs = sorted(glob.glob(model_pattern), reverse=True)
            
            if not matching_dirs:
                raise ValueError(f"No saved models found for {Signal_Directory}")
            
            model_base_dir = matching_dirs[0]
            print(f"Loading most recent model: {os.path.basename(model_base_dir)}")
        else:
            # Use the specified model name
            model_base_dir = os.path.join(models_base_path, model_name)
            
            if not os.path.exists(model_base_dir):
                raise ValueError(f"Model directory not found: {model_base_dir}")
            
            print(f"Loading specified model: {model_name}")
        
        # Load metadata to determine model structure
        try:
            with open(os.path.join(model_base_dir, "metadata.json"), "r") as f:
                metadata = json.load(f)
                
            # Update parameters based on metadata
            n_ensemble_sets = metadata.get('n_ensemble_sets', n_ensemble_sets)
            bdt_models_per_set = metadata.get('bdt_models_per_set', bdt_models_per_set)
            
            print(f"Using model configuration: {n_ensemble_sets} ensemble sets")
        except Exception as e:
            print(f"Warning: Failed to load metadata - {e}")
    
    else:
        raise ValueError("Mode must be either 'train' or 'load'")
    
    # Create comparison directory for plots
    comparison_dir = os.path.join(plot_directory, f'ensemble_comparison_{Signal_Directory}_{signal_injection_ratio}_{time_str}')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # ===== PREPARE DATA =====
    if mode == "train":
        # Prepare data splits
        data_splits = prepare_data_splits(
            sig, bkg1, 
            test_fraction=test_fraction,
            signal_injection_ratio=signal_injection_ratio,
            random_seed=random_seed,
            m_tt_min=m_tt_min,
            m_tt_max=m_tt_max,
            column_labels=column_labels,
            **kwargs
        )
        
        # Extract test data
        test_data = data_splits['test_data']
            
        columns_to_drop = ["m_tau1tau2"] if "m_tau1tau2" in test_data.columns else []
        if columns_to_drop:
            test_data = test_data.drop(columns=columns_to_drop)
        
        X_test = test_data.iloc[:, 0:-1].values
        y_test = test_data['label'].values
        
        # Remove NaN values
        nan_mask = ~np.isnan(X_test).any(axis=1)
        X_test = X_test[nan_mask]
        y_test = y_test[nan_mask]
        
        # Save test data for later use
        os.makedirs(os.path.join(model_base_dir, "test_data"), exist_ok=True)
        np.save(os.path.join(model_base_dir, "test_data", "X_test.npy"), X_test)
        np.save(os.path.join(model_base_dir, "test_data", "y_test.npy"), y_test)
        
        # Prepare samples
        sample_columns = column_labels[:n_dim+1]
        samples_df = pd.DataFrame(samples[:, :n_dim+1], columns=sample_columns)
        print("samples_df shape:", samples_df.shape)
        
        columns_to_drop = ["Background_number_of_objects"] if "Background_number_of_objects" in samples_df.columns else []
        if columns_to_drop:
            samples_df = samples_df.drop(columns=columns_to_drop)
        
        columns_to_drop = ["labels","m_tau1tau2"] if "labels" in samples_df.columns else []
        if columns_to_drop:
            samples_df = samples_df.drop(columns=columns_to_drop)
        
        columns_to_drop = ["m_tau1tau2"] if "m_tau1tau2" in samples_df.columns else []
        if columns_to_drop:
            samples_df = samples_df.drop(columns=columns_to_drop)
        
        samples_df['label'] = 0
        
    else:  # mode == "load"
        # Load the saved test data
        try:
            X_test = np.load(os.path.join(model_base_dir, "test_data", "X_test.npy"))
            y_test = np.load(os.path.join(model_base_dir, "test_data", "y_test.npy"))
            
            print(f"Loaded test data: {X_test.shape[0]} samples with {X_test.shape[1]} features")
        except Exception as e:
            print(f"Error loading test data: {e}")
            # If we can't load the test data, raise an error
            raise ValueError("Test data not found and not provided")
            
    # ===== TRAIN BDT ENSEMBLE SETS =====
    bdt_ensemble_sets = []
    bdt_ensemble_preds = []
    bdt_ensemble_aucs = []
    bdt_meta_pred = None
    bdt_meta_auc = None
    
    if use_bdt:
        # Create a list to store BDT models
        bdt_models_list = []
        
        print("\n===== Training BDT Ensemble Models =====")
        
        if mode == "train":
            # Train multiple BDT models with different train/validation splits
            for set_idx in range(bdt_models_per_set):
                print(f"\n----- Training BDT Ensemble Set {set_idx+1}/{bdt_models_per_set} -----")
                
                # Create a different seed for each ensemble set
                set_seed = random_seed + (set_idx * 100)
                np.random.seed(set_seed)
                
                # Select random signal samples for this ensemble set
                signal_indices = np.random.permutation(len(data_splits['signal_remaining']))[:data_splits['n_signal_to_inject']]
                signal_for_ensemble = data_splits['signal_remaining'].iloc[signal_indices].copy()
                
                # Create real data with injected signal (label=1)
                background_train = data_splits['background_train'].copy()
                real_data = pd.concat([background_train, signal_for_ensemble], axis=0).reset_index(drop=True)
                
                # Clean up columns and set label
                columns_to_drop = ["Background_number_of_objects", "labels", "m_tau1tau2"]
                for col in columns_to_drop:
                    if col in real_data.columns:
                        real_data = real_data.drop(columns=[col])
                real_data['label'] = 1
                
                # Shuffle real data
                real_data = real_data.sample(frac=1, random_state=set_seed).reset_index(drop=True)
                
                # Get CATHODE-generated samples with label 0
                gen_samples = samples_df.sample(n=len(real_data), random_state=set_seed).reset_index(drop=True)
                gen_samples['label'] = 0
                
                # Combine data
                ensemble_train_data = pd.concat([real_data, gen_samples], axis=0).reset_index(drop=True)
                ensemble_train_data = ensemble_train_data.sample(frac=1, random_state=set_seed)
                
                # Create train/val split
                X_train_full = ensemble_train_data.iloc[:, 0:-1].values
                y_train_full = ensemble_train_data['label'].values
                
                # Create train/val split for this model
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_full, y_train_full, 
                    test_size=Train_val_split, 
                    random_state=set_seed,
                    stratify=y_train_full
                )
                
                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Create and train BDT model
                bdt_model = HGBClassifier(
                    save_path=None,  # Don't save individual models
                    early_stopping=True,
                    max_iters=400,
                    verbose=False
                )
                
                bdt_model.fit(X_train_scaled, y_train, X_val_scaled, y_val)
                
                # Create pipeline with scaler and model
                from sklearn.pipeline import make_pipeline
                model = make_pipeline(scaler, bdt_model)
                
                # Add model to list
                bdt_models_list.append(model)
        
        elif mode == "load":
            # Load saved models if available
            models_path = os.path.join(model_base_dir, "bdt_models")
            if os.path.exists(models_path):
                for i in range(bdt_models_per_set):
                    model_file = os.path.join(models_path, f"bdt_model_{i}.pkl")
                    if os.path.exists(model_file):
                        with open(model_file, 'rb') as f:
                            model = pickle.load(f)
                        bdt_models_list.append(model)
                    else:
                        print(f"Warning: Model file {model_file} not found")
            else:
                print(f"Warning: Models directory {models_path} not found")
        
        # Create the ensemble model if models were loaded or trained
        if bdt_models_list:
            bdt_ensemble = EnsembleModel(bdt_models_list)
            
            # Make predictions
            bdt_meta_pred = bdt_ensemble.predict(X_test)
            bdt_meta_auc = roc_auc_score(y_test, bdt_meta_pred)
            
            print(f"\nBDT Ensemble AUC: {bdt_meta_auc:.4f}")
        else:
            print("No BDT models available for ensemble prediction")
            
    # ===== STEP 5: CREATE COMPARISON PLOTS =====
    bdt_fpr, bdt_tpr, bdt_sic = None, None, None
    bdt_fpr_001, bdt_sic_001 = None, None

    if use_bdt and bdt_meta_pred is not None:
        # Calculate ROC and SIC for BDT meta-ensemble
        bdt_fpr, bdt_tpr, _ = roc_curve(y_test, bdt_meta_pred)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            bdt_sic = bdt_tpr / np.sqrt(bdt_fpr)
            bdt_sic = np.where(bdt_fpr < 1e-10, 0, bdt_sic)  # Handle division by zero
            bdt_sic = np.nan_to_num(bdt_sic, nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaNs and infs
        
        # Find SIC at FPR=0.01 for BDT
        bdt_fpr_001_idx = np.argmin(np.abs(bdt_fpr - 0.01))
        bdt_sic_001 = bdt_sic[bdt_fpr_001_idx]
        bdt_fpr_001 = bdt_fpr[bdt_fpr_001_idx]
        bdt_tpr_001 = bdt_tpr[bdt_fpr_001_idx]
        
        print(f"BDT Meta-Ensemble SIC@FPR≈0.01: {bdt_sic_001:.2f} (FPR={bdt_fpr_001:.4f}, TPR={bdt_tpr_001:.4f})")
        
        # Store results
        results['bdt'] = {
            'meta_pred': bdt_meta_pred,
            'meta_auc': bdt_meta_auc,
            'ensemble_preds': bdt_ensemble_preds,
            'ensemble_aucs': bdt_ensemble_aucs,
            'fpr': bdt_fpr,
            'tpr': bdt_tpr,
            'sic': bdt_sic,
            'sic_001': bdt_sic_001,
            'fpr_001': bdt_fpr_001,
            'tpr_001': bdt_tpr_001
        }

        # ROC Curve for BDT
        plt.figure(figsize=(12, 10))
        plt.plot(bdt_fpr, bdt_tpr, color='cyan', linewidth=3, 
                label=f"BDT Meta-Ensemble (AUC={bdt_meta_auc:.3f})")
        plt.plot([0, 1], [0, 1], "w:", label="Random", linewidth=1.5)
        plt.xlim([0, 1.0])
        plt.ylim([0, 1.0])
        plt.xlabel("False Positive Rate (FPR)", fontsize=14)
        plt.ylabel("True Positive Rate (TPR)", fontsize=14)
        plt.title(f"BDT ROC Curve (Signal Injection: {signal_injection_ratio:.1%})", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right", fontsize=12)
        plt.savefig(os.path.join(comparison_dir, "bdt_roc.png"))
        plt.close()
        
        # SIC vs FPR Curve for BDT
        plt.figure(figsize=(12, 10))
        random_tpr = np.linspace(0.001, 1, 1000)
        random_sic = random_tpr / np.sqrt(random_tpr)
        plt.plot(random_tpr, random_sic, "w:", label="Random", linewidth=1.5)
        plt.plot(bdt_fpr, bdt_sic, color='cyan', linewidth=3,
                label=f"BDT Meta-Ensemble (AUC={bdt_meta_auc:.3f})")
        plt.xscale('log')
        plt.xlim([0.001, 1.0])
        plt.ylim([0, np.nanmax(bdt_sic) * 1.1])
        plt.xlabel("Background Efficiency (FPR)", fontsize=14)
        plt.ylabel("Significance Improvement", fontsize=14)
        plt.title(f"BDT SIC Curve (Signal Injection: {signal_injection_ratio:.1%})", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc="best")
        plt.savefig(os.path.join(comparison_dir, "bdt_sic_fpr.png"))
        plt.close()
    
    # Record end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nTotal runtime: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    
    # Return results with model directory
    results['model_dir'] = model_base_dir
    results['comparison_dir'] = comparison_dir
    
    return results, comparison_dir