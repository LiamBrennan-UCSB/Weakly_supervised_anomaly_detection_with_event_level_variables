"""
Phase space processing functions for CATHODE analysis.
"""
import os
import numpy as np
import torch
from torch.multiprocessing import Pool
from functools import partial

from config import *
from utils.Processor import PSDistanceMaker, process_chunks, PhaseSpace_boost_standalone_torch

def process_background_data(file_paths):
    """Process background data from ROOT files."""
    background_data_dictionary = {}
    
    # Basic event kinematic variables we want to extract
    background_variable_names = [
        'Background_TwoTau_mjj', 'Background_TwoTau_deltaR','Background_Event_Type', 
        'Background_Sum_Other_Jet_Pt', 'Background_Sum_MET_pt'
    ]

    # Create arrays for each variable
    background_arrays = {var_name: [] for var_name in background_variable_names}
    
    for background_file in file_paths['background_files']:
        print(f"Processing {background_file}, max_evt_bkg: {MAX_BACKGROUND_EVENTS}")
        bkg_file = f'{ROOT_FILE_BASE_DIRECTORY}/{background_file}'
        
        # Create PSDistanceMaker to process the file
        psdm = PSDistanceMaker(
            file_paths['signal_data_path'], 
            bkg_file, 
            n_dim=N_DIMS, 
            max_Signal_Events=MAX_SIGNAL_EVENTS,
            jet_type=JET_TYPE,
            max_Background_Events=MAX_BACKGROUND_EVENTS, 
        )

        psdm.PhaseSpace_load(TURN_ON_BREAK_COUNT, NUMBER_TO_PAD_TO, 'Background')

        # Store kinematic variables
        for var_name in background_variable_names:
            if hasattr(psdm, var_name):
                background_arrays[var_name].append(np.array(getattr(psdm, var_name)))
            else:
                background_arrays[var_name].append(np.array([]))

    # Concatenate arrays from multiple files if needed
    for key in background_arrays:
        valid_arrays = [arr for arr in background_arrays[key] if arr.size > 0]
        if valid_arrays:
            background_arrays[key] = np.concatenate(valid_arrays)
        else:
            background_arrays[key] = np.array([])

    # Add arrays to dictionary
    background_data_dictionary.update(background_arrays)

    return background_data_dictionary


def process_signal_data(file_paths, background_data):
    """Process signal data from ROOT files."""
    psdm = PSDistanceMaker(
        file_paths['signal_data_path'], 
        file_paths['bkg_data_path'], 
        n_dim=N_DIMS, 
        max_Signal_Events=MAX_SIGNAL_EVENTS,
        jet_type=JET_TYPE,
        max_Background_Events=MAX_BACKGROUND_EVENTS, 
    )

    psdm.PhaseSpace_load(TURN_ON_BREAK_COUNT, NUMBER_TO_PAD_TO, 'Signal')

    # Extract kinematic variables
    variable_names = [
        'Signal_TwoTau_mjj', 'Signal_TwoTau_deltaR', 'Signal_Event_Type',
        'Signal_Sum_Other_Jet_Pt', 'Signal_Sum_MET_pt'
    ]

    signal_arrays = {}
    for var_name in variable_names:
        if hasattr(psdm, var_name):
            signal_arrays[var_name] = np.array(getattr(psdm, var_name))
        else:
            # If the variable doesn't exist, store an empty array
            signal_arrays[var_name] = np.array([])

    # Create the signal data dictionary
    signal_data_dictionary = {}
    signal_data_dictionary.update(signal_arrays)

    return signal_data_dictionary


def prepare_datasets(signal_data, background_data):
    """Convert signal and background data into datasets for analysis."""
    from utils.Naming_Conventions import Simplified_Naming_Convention
    
    # Create datasets using the simplified naming convention function
    datasets = Simplified_Naming_Convention(
        signal_data, background_data
    )
    
    # Unpack the results
    dataset_sig, dataset_bg, _, _, _, column_labels_bkg, _, Training_events, Validation_events, Test_events = datasets
    
    return {
        'dataset_sig': dataset_sig,
        'dataset_bg': dataset_bg,
        'column_labels': column_labels_bkg,
        'Training_events': Training_events,
        'Validation_events': Validation_events,
        'Test_events': Test_events
    }