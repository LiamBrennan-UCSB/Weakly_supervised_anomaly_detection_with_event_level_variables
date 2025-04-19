"""
Data splitting and preparation utilities for CATHODE analysis.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle


def make_train_test_val_simplified(sig, bkg1, column_labels=None, group_by_object_count=False, 
                                 m_tt_min=0.175, m_tt_max=0.3, 
                                 sig_injection=0.01, testing_set_size=0.7, 
                                 random_seed=42, **kwargs):
    """
    Create training, validation, and test datasets for CATHODE analysis with signal injection.
    
    Parameters:
    -----------
    sig : numpy.ndarray or pandas.DataFrame
        Signal dataset
    bkg1 : numpy.ndarray or pandas.DataFrame
        Primary background dataset
    column_labels : list, optional
        Column labels for dataframes if using numpy arrays
    group_by_object_count : bool, default=False
        Whether to group data by object count
    m_tt_min : float, default=0.175
        Minimum mass for signal region
    m_tt_max : float, default=0.3
        Maximum mass for signal region
    sig_injection : float, default=0.01
        Signal injection ratio for training
    testing_set_size : float, default=0.7
        Fraction of data to use for testing
    random_seed : int, default=42
        Random seed for reproducibility
    **kwargs : dict
        Additional keyword arguments
        
    Returns:
    --------
    tuple : Various datasets and metadata needed for analysis
    """
    # Convert numpy arrays to pandas DataFrames if needed
    if isinstance(sig, np.ndarray) and column_labels is not None:
        sig = pd.DataFrame(sig, columns=column_labels)
        bkg1 = pd.DataFrame(bkg1, columns=column_labels)
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        import random
        random.seed(random_seed)

    # First, prepare the data splits
    data_splits = prepare_data_splits(sig, bkg1, 
                                     test_fraction=testing_set_size,
                                     signal_injection_ratio=sig_injection,
                                     random_seed=random_seed,
                                     m_tt_min=m_tt_min,
                                     m_tt_max=m_tt_max,
                                     column_labels=column_labels,
                                     **kwargs)
    
    # Extract datasets
    background_train = data_splits['background_train']
    signal_remaining = data_splits['signal_remaining']
    n_signal_to_inject = data_splits['n_signal_to_inject']
    test_data = data_splits['test_data']
    sideband_train = data_splits['sideband_train']
    sideband_val = data_splits['sideband_val']
    
    # Create training set with signal injection
    signal_for_training = signal_remaining.iloc[:n_signal_to_inject].copy()
    signal_for_training['label'] = 1  # Signal truth label
    
    # Combine background and injected signal for training
    train_data = pd.concat([background_train, signal_for_training], axis=0).reset_index(drop=True)
    train_data = train_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)  # Shuffle
    
    # Create validation set (use remainder of signal for validation)
    remaining_signal_for_val = signal_remaining.iloc[n_signal_to_inject:].copy()
    remaining_signal_for_val['label'] = 1  # Signal truth label
    
    # Take a portion of background for validation
    val_size = min(len(remaining_signal_for_val), int(len(background_train) * 0.25))
    background_val = background_train.sample(n=val_size, random_state=random_seed).copy()
    
    # Combine for validation set
    val_data = pd.concat([background_val, remaining_signal_for_val], axis=0).reset_index(drop=True)
    val_data = val_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)  # Shuffle
    
    # Prepare feature list
    feature_list = column_labels[:-1] if column_labels else [f"feature_{i}" for i in range(train_data.shape[1]-1)]
    
    # Additional datasets required by the CATHODE framework
    # Creating outer and inner data based on signal and background regions
    outerdata_train = train_data[train_data['label'] == 0].iloc[:, :-1].values  # Background only
    outerdata_val = val_data[val_data['label'] == 0].iloc[:, :-1].values  # Background only
    
    innerdata_train = train_data.iloc[:, :-1].values  # All training data (bg + signal)
    innerdata_val = val_data.iloc[:, :-1].values  # All validation data (bg + signal)
    innerdata_test = test_data.iloc[:, :-1].values  # All test data
    
    # Create a Set_aside_background dataset
    Set_aside_background = sideband_train.iloc[:, :-1].values if len(sideband_train) > 0 else np.zeros((0, train_data.shape[1]-1))
    
    # Convert final dataframes to numpy arrays for consistency
    train = train_data.values
    val = val_data.values
    test = test_data.values
    
    # Create placeholder tensors for compatibility with existing code
    All_objects_in_events = torch.zeros((1, 10))  # Placeholder tensor
    Training_events = torch.zeros((1, 10))  # Placeholder tensor
    Validation_events = torch.zeros((1, 10))  # Placeholder tensor
    Test_events = torch.zeros((1, 10))  # Placeholder tensor
    
    return (train, val, test, feature_list, 
            outerdata_train, outerdata_val, innerdata_train, innerdata_val, innerdata_test,
            Set_aside_background)


def prepare_data_splits(sig, bkg1, test_fraction=0.25, signal_injection_ratio=0.01, random_seed=None,
                       m_tt_min=0.175, m_tt_max=0.3, column_labels=None, **kwargs):
    """
    Prepare data splits for testing and training with proper signal injection.
    
    Parameters:
    -----------
    sig : pandas.DataFrame
        Signal dataset
    bkg1 : pandas.DataFrame
        Background dataset
    test_fraction : float, default=0.25
        Fraction of data to use for testing
    signal_injection_ratio : float, default=0.01
        Signal injection ratio
    random_seed : int, optional
        Random seed for reproducibility
    m_tt_min : float, default=0.175
        Minimum mass for signal region
    m_tt_max : float, default=0.3
        Maximum mass for signal region
    column_labels : list, optional
        Column labels for dataframes
    **kwargs : dict
        Additional keyword arguments
        
    Returns:
    --------
    dict : Various datasets and metadata
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        import random
        random.seed(random_seed)

    # Ensure column labels are assigned if provided
    if column_labels is not None:
        sig.columns = column_labels
        bkg1.columns = column_labels
    
    # Split data by mass region
    sig_sigregion = sig[(sig['m_tau1tau2'] >= m_tt_min) & (sig['m_tau1tau2'] < m_tt_max)]
    sig_bkgregion = sig[~((sig['m_tau1tau2'] >= m_tt_min) & (sig['m_tau1tau2'] < m_tt_max))]
    
    bkg1_sigregion = bkg1[(bkg1['m_tau1tau2'] >= m_tt_min) & (bkg1['m_tau1tau2'] < m_tt_max)]
    bkg1_bkgregion = bkg1[~((bkg1['m_tau1tau2'] >= m_tt_min) & (bkg1['m_tau1tau2'] < m_tt_max))]
    
    # Define background signal region and sideband
    background_signal_region = bkg1_sigregion
    background_sideband = bkg1_bkgregion
    
    # Shuffle all datasets for randomization
    background_signal_region = background_signal_region.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    sig_sigregion = sig_sigregion.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    background_sideband = background_sideband.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Print dataset sizes
    print(f"Total background in signal region: {len(background_signal_region)}")
    print(f"Total signal available: {len(sig_sigregion)}")
    
    # STEP 1: Create balanced test set
    # Calculate test set size
    test_bg_size = int(len(background_signal_region) * test_fraction)
    print(f"Setting aside {test_bg_size} background events for testing")
    
    # Split background into test and train
    background_test = background_signal_region.iloc[:test_bg_size].copy()
    background_train = background_signal_region.iloc[test_bg_size:].copy()
    
    # Determine how many signal events to use for testing (equal to background test size)
    if len(sig_sigregion) < test_bg_size:
        print(f"Warning: Not enough signal events for balanced testing. Using all {len(sig_sigregion)} available.")
        signal_test_size = len(sig_sigregion)
    else:
        signal_test_size = test_bg_size
    
    # Split signal into test and remaining pool
    signal_test = sig_sigregion.iloc[:signal_test_size].copy()
    signal_remaining = sig_sigregion.iloc[signal_test_size:].copy()
    
    # Set truth labels for test sets
    background_test['label'] = 0  # Background truth label
    signal_test['label'] = 1      # Signal truth label
    
    # Combine for balanced test set
    test_data = pd.concat([background_test, signal_test], axis=0).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)  # Shuffle
    
    # STEP 2: Calculate signal injection amount for training
    n_background_train = len(background_train)
    
    # Signal fraction = signal_events / total_events
    # where total_events = background_events + signal_events
    # Solving for signal_events:
    # signal_events = (signal_fraction * background_events) / (1 - signal_fraction)
    n_signal_to_inject = int((signal_injection_ratio * n_background_train) / (1 - signal_injection_ratio))

    if n_signal_to_inject > len(signal_remaining):
        print(f"Warning: Not enough signal events for requested injection rate. Using all {len(signal_remaining)} available.")
        n_signal_to_inject = len(signal_remaining)
    
    print(f"Injection amount: {n_signal_to_inject} signal events ({signal_injection_ratio:.1%} of {n_background_train} background events)")
    
    # Set labels for background
    background_train['label'] = 0  # Background truth label
    
    # Prepare sideband data (typically split 80/20)
    sideband_train_idx = int(0.8 * len(background_sideband))
    sideband_train = background_sideband.iloc[:sideband_train_idx].copy()
    sideband_val = background_sideband.iloc[sideband_train_idx:].copy()
    
    # Set labels for sideband data (all background)
    if len(sideband_train) > 0:
        sideband_train['label'] = 0
    if len(sideband_val) > 0:
        sideband_val['label'] = 0
    
    # Return all the data splits
    return {
        'background_train': background_train,
        'signal_remaining': signal_remaining,
        'n_signal_to_inject': n_signal_to_inject,
        'test_data': test_data,
        'sideband_train': sideband_train,
        'sideband_val': sideband_val,
        'sig_sigregion': sig_sigregion,
        'background_signal_region': background_signal_region
    }