"""
Event data processing functions for CATHODE analysis.
"""
import numpy as np
import torch
from torch.multiprocessing import Pool
from functools import partial

from config import *
from utils.PhaseSpaceDistanceMaker import PSDistanceMaker


def process_background_data(file_paths):
    """Process background data from ROOT files."""
    background_data_dictionary = {}
    
    background_variable_names = [ 
        'Background_TwoTau_mjj', 'Background_TwoTau_deltaR', 'Background_Event_Type',
        'Background_Sum_Non_Tau_Jet_Pt', 'Background_Sum_MET_pt'
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
            file_paths['extra_bkg_data_path'],
            tree_name='Events',
            n_dim=N_DIMS, 
            max_nevents=MAX_SIGNAL_EVENTS,
            jet_type=JET_TYPE,
            max_evt_bkg=MAX_BACKGROUND_EVENTS
        )

        # Load the background data
        psdm.PhaseSpace_load(TURN_ON_BREAK_COUNT, NUMBER_TO_PAD_TO, 'Background')

        # Store the raw kinematic variables
        for var_name in background_variable_names:
            if hasattr(psdm, var_name):
                background_arrays[var_name].append(np.array(getattr(psdm, var_name)))
            else:
                # If the variable doesn't exist, store an empty array
                background_arrays[var_name].append(np.array([]))

    # Concatenate arrays from multiple files if needed
    for key in background_arrays:
        valid_arrays = [arr for arr in background_arrays[key] if arr.size > 0]
        if valid_arrays:
            background_arrays[key] = np.concatenate(valid_arrays)
        else:
            background_arrays[key] = np.array([])

    # Add all arrays to the final dictionary
    background_data_dictionary.update(background_arrays)

    return background_data_dictionary


def process_signal_data(file_paths, background_data):
    """Process signal data from ROOT files."""
    # Create PSDistanceMaker to process the file
    psdm = PSDistanceMaker(
        file_paths['signal_data_path'], 
        file_paths['bkg_data_path'], 
        file_paths['extra_bkg_data_path'],
        tree_name='Events',
        n_dim=N_DIMS, 
        max_nevents=MAX_SIGNAL_EVENTS,
        jet_type=JET_TYPE,
        max_evt_bkg=MAX_BACKGROUND_EVENTS
    )

    # Load the signal data
    psdm.PhaseSpace_load(TURN_ON_BREAK_COUNT, NUMBER_TO_PAD_TO, 'Signal')

    variable_names = [
        'Signal_TwoTau_mjj', 'Signal_TwoTau_deltaR', 'Signal_Event_Type', 'Signal_Sum_Non_Tau_Jet_Pt', 'Signal_Sum_MET_pt'
    ]

    signal_arrays = {}
    for var_name in variable_names:
        if hasattr(psdm, var_name):
            signal_arrays[var_name] = np.array(getattr(psdm, var_name))
        else:
            # If the variable doesn't exist, store an empty array
            signal_arrays[var_name] = np.array([])

    # Create the signal data dictionary with the basic arrays
    signal_data_dictionary = {}
    signal_data_dictionary.update(signal_arrays)
    
    for var_name in kinematic_variable_names:
        signal_data_dictionary[var_name] = np.array([])

    return signal_data_dictionary


def prepare_datasets(signal_data, background_data):
    """
    Convert signal and background data into datasets for analysis.
    This is a simplified version that bypasses the phase space calculation
    and provides compatibility with the rest of the pipeline.
    """
    from utils.Naming_Conventions import Simplified_Naming_Convention
    
    # Create datasets using the simplified naming convention function
    datasets = Simplified_Naming_Convention(
        signal_data, background_data
    )
    
    # Unpack the results
    dataset_sig, dataset_bg, dataset_extrabkg, dataset_extrabkg_2, dataset_extrabkg_3, \
    column_labels_bkg, All_objects_in_events, Training_events, Validation_events, \
    Test_events = datasets
    
    return {
        'dataset_sig': dataset_sig,
        'dataset_bg': dataset_bg,
        'column_labels_bkg': column_labels_bkg,
        'Training_events': Training_events,
        'Validation_events': Validation_events,
        'Test_events': Test_events
    }