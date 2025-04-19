import os
import numpy as np
import torch
import ROOT
from torch.multiprocessing import Pool
from functools import partial

class PSDistanceMaker(object):
    """
    Simplified version of PSDistanceMaker that focuses only on basic kinematic variables:
    - Sum_MET_pt
    - Sum_Other_Jet_pt
    - Delta_R
    - mjj
    - Event_Type (for background)
    
    Phase Space calculations have been removed.
    """
    
    def __init__(self, sig_file, bkg_file, tree_name='Events', n_dim=2, max_Signal_Events=5000, max_Background_Events=5000, suffix='_total', jet_type='Tau_jet'):
        """Initialize the PSDistanceMaker with file paths and configuration parameters."""
        self.sig_file = sig_file
        self.bkg_file = bkg_file

        self.tree_name = tree_name
        self.jet_type = jet_type
        self.n_dim = n_dim
        self.max_Signal_Events = max_Signal_Events
        self.suffix = suffix
        self.max_Background_Events = max_Background_Events

    def PhaseSpace_load(self, turn_on_break_count, number_to_pad_to, signal_or_bkg):
        """Load the required data from ROOT files."""
        # Open ROOT files
        f_sig = ROOT.TFile.Open(self.sig_file)
        f_bkg = ROOT.TFile.Open(self.bkg_file)
        # Get trees
        self.tree_sig = f_sig.Get(self.tree_name)
        self.tree_bkg = f_bkg.Get(self.tree_name)
        
        # Process the appropriate data based on signal_or_bkg parameter
        if signal_or_bkg == 'Signal':
            # Extract only the key variables we need
            self.Signal_TwoTau_mjj, self.Signal_TwoTau_deltaR, self.Signal_Event_Type, \
            self.Signal_Sum_Other_Jet_Pt, self.Signal_Sum_MET_pt = self._extract_base_variables(
                self.tree_sig, 'Signal', turn_on_break_count)
            
            # Store event 4-vectors for further processing if needed
            self.event_pN_sig = self.__loadEvent_pN(
                self.tree_sig, label=1, Bkg_or_signal='Signal', turn_on_break_count=turn_on_break_count,
                number_to_pad_to=number_to_pad_to)
            
        elif signal_or_bkg == 'Background':
            # Extract only the key variables we need
            self.Background_TwoTau_mjj, self.Background_TwoTau_deltaR, self.Background_Event_Type, \
            self.Background_Sum_Other_Jet_Pt, self.Background_Sum_MET_pt = self._extract_base_variables(
                self.tree_bkg, 'Background', turn_on_break_count)
            
            # Store event 4-vectors for further processing if needed
            self.event_pN_bkg = self.__loadEvent_pN(
                self.tree_bkg, label=0, Bkg_or_signal='Background',turn_on_break_count=turn_on_break_count,number_to_pad_to=number_to_pad_to)
            
            
    def _extract_base_variables(self, tree, data_type, turn_on_break_count):
        """
        Extract basic kinematic variables from a ROOT tree.
        
        Returns:
        - mjj: Invariant mass of the two tau jets
        - deltaR: Delta R between the two tau jets
        - event_type: Event type (background only)
        - sum_other_jet_pt: Sum pT of non-tau jets
        - sum_met_pt: Sum pT of MET
        """
        mjj_list = []
        deltaR_list = []
        event_type_list = []
        sum_other_jet_pt_list = []
        sum_met_pt_list = []
        
        max_events = self._get_max_events(data_type)
        event_count = 0
        
        for event in tree:
            # Check if we've reached the maximum number of events
            if turn_on_break_count and event_count >= max_events:
                break
                
            # Check if the event has at least two tau jets
            tau_jet_count = len(getattr(event, 'Tau_Jet_pt'))
            if tau_jet_count < 2:
                continue
                
            # Skip events for pure background as needed
            if data_type == 'bkg':
                event_count += 1
                continue
                
            # Extract event type
            if hasattr(event, 'event_type') and len(getattr(event, 'event_type')) > 0:
                event_type = getattr(event, 'event_type')[0]
                event_type_list.append(event_type)
            else:
                event_type_list.append(0)  # Default value
                
            # Extract mjj
            if hasattr(event, 'm_tau1tau2') and len(getattr(event, 'm_tau1tau2')) > 0:
                mjj = getattr(event, 'm_tau1tau2')[0]
                mjj_list.append(mjj)
            else:
                mjj_list.append(0)  # Default value
                
            # Calculate deltaR between top two tau jets
            if tau_jet_count >= 2:
                # Get phi and eta values
                tau_phi = getattr(event, 'Tau_Jet_phi')
                tau_eta = getattr(event, 'Tau_Jet_eta')
                tau_pt = getattr(event, 'Tau_Jet_pt')
                
                # Sort by pT to get top two jets
                sorted_indices = sorted(range(tau_jet_count), key=lambda i: tau_pt[i], reverse=True)
                idx1, idx2 = sorted_indices[0], sorted_indices[1]
                
                # Calculate deltaR
                delta_phi = abs(tau_phi[idx1] - tau_phi[idx2])
                if delta_phi > np.pi:
                    delta_phi = 2 * np.pi - delta_phi
                    
                delta_eta = abs(tau_eta[idx1] - tau_eta[idx2])
                delta_r = np.sqrt(delta_phi**2 + delta_eta**2)
                deltaR_list.append(delta_r)
            else:
                deltaR_list.append(0)  # Default value
                
            # Calculate sum of non-tau jet pT
            non_tau_jet_pt = getattr(event, 'Non_Tau_Jet_pt')
            sum_other_jet_pt = sum(non_tau_jet_pt)
            
            # Add small noise to zero sums to match original implementation
            if sum_other_jet_pt == 0:
                if np.random.random() < 0.6:
                    beta_val = np.random.beta(2.5, 1)
                    noise = 21 * beta_val
                    sum_other_jet_pt += noise
                else:
                    noise = np.random.normal(loc=20.0, scale=10, size=1)[0]
                    sum_other_jet_pt += noise
                    
            sum_other_jet_pt_list.append(sum_other_jet_pt)
            
            # Calculate sum of MET pT
            met_pt = getattr(event, 'met_pt')
            sum_met_pt = sum(met_pt)
            sum_met_pt_list.append(sum_met_pt)
            
            event_count += 1
            
        return mjj_list, deltaR_list, event_type_list, sum_other_jet_pt_list, sum_met_pt_list

    def _get_max_events(self, data_type):
        """Get the maximum number of events to process based on data type."""
        if data_type == 'Signal':
            return self.max_Signal_Events
        elif data_type == 'Background':
            return self.max_Background_Events
        else:
            return 1000  # Default value
            
    def __loadEvent_pN(self, tree, label, Bkg_or_signal, turn_on_break_count=True, 
                      number_to_pad_to=15):
        """
        Load event 4-momentum vectors data from a ROOT tree.
        This is a simplified version that retains only the essential functionality.
        
        Returns:
        Dictionary of the form {(label, event_id): numpy.ndarray of 4-vectors}
        """
        event_pN_dict = {}
        count = 0
        
        for event_idx, event in enumerate(tree):
            # Get basic event properties
            tau_jet_count = len(getattr(event, 'Tau_Jet_pt'))
            tau_jets_exist = tau_jet_count >= self.n_dim
            
            # Process events with tau jets
            if tau_jets_exist:
                energy_, px_, py_, pz_ = self._collect_tau_and_other_objects(
                    event
                )
                
                amount_to_pad = number_to_pad_to - tau_jet_count
            else:
                # Skip this event as it doesn't meet our criteria
                continue
            
            # Apply zero padding
            for _ in range(amount_to_pad):
                energy_.append(0.1)  # Small non-zero value
                px_.append(0.1)
                py_.append(0)
                pz_.append(0)
            
            if turn_on_break_count:
                if self._should_break_eventpN(Bkg_or_signal, count):
                    break
            
            # Create event 4-vector array
            pN_evt = self.__loadpN(energy_, px_, py_, pz_, number_to_pad_to)
                
            count += 1
            
            # Add to appropriate dictionary based on event type
            if Bkg_or_signal == 'Background':
                event_pN_dict[(label, count)] = pN_evt
            elif Bkg_or_signal == 'Signal':
                event_pN_dict[(label, count)] = pN_evt
                
        return event_pN_dict

    def _should_break_eventpN(self, Bkg_or_signal, count):
        """Determine if we should break the event loop based on event counts"""
        if Bkg_or_signal == "Signal" and count >= self.max_Signal_Events:
            return True
        if Bkg_or_signal == "Background" and count >= self.max_Background_Events:
            return True
        return False

    def _count_non_tau_objects(self, event):
        """Count the non-tau objects in an event based on enabled flags"""
        count = 0
        count += len(getattr(event, 'Non_Tau_Jet_pt'))
        count += len(getattr(event, 'Isolated_Electrons_pt'))
        count += len(getattr(event, 'Isolated_Muons_pt'))
        count += len(getattr(event, 'Isolated_Photons_pt'))
        return count

    def _collect_non_tau_objects(self, event):
        """Collect non-tau objects from an event based on enabled flags"""
        energy_, px_, py_, pz_ = [], [], [], []
        
        self._add_non_tau_jets(event, energy_, px_, py_, pz_)
        self._add_electrons(event, energy_, px_, py_, pz_)
        self._add_muons(event, energy_, px_, py_, pz_)
        self._add_photons(event, energy_, px_, py_, pz_)
                
        return energy_, px_, py_, pz_

    def _collect_tau_and_other_objects(self, event):
        """Collect tau and other objects from an event based on enabled flags"""
        energy_, px_, py_, pz_ = [], [], [], []
        
        self._add_tau_jets(event, energy_, px_, py_, pz_)
        
        # Add all other object types
        other_energy, other_px, other_py, other_pz = self._collect_non_tau_objects(event)
        
        # Combine all vectors
        energy_.extend(other_energy)
        px_.extend(other_px)
        py_.extend(other_py)
        pz_.extend(other_pz)
        
        return energy_, px_, py_, pz_

    def _add_non_tau_jets(self, event, energy_, px_, py_, pz_):
        """Add non-tau jets to the 4-vector lists"""
        px_non_tau_Jet = getattr(event, 'Non_Tau_Jet_px')
        py_non_tau_Jet = getattr(event, 'Non_Tau_Jet_py')
        pz_non_tau_Jet = getattr(event, 'Non_Tau_Jet_pz')
        
        for px, py, pz in zip(px_non_tau_Jet, py_non_tau_Jet, pz_non_tau_Jet):
            px_.append(px)
            py_.append(py)
            pz_.append(pz)
            energy = np.sqrt(px**2 + py**2 + pz**2)
            energy_.append(energy)

    def _add_electrons(self, event, energy_, px_, py_, pz_):
        """Add electrons to the 4-vector lists"""
        px_electron = getattr(event, 'Isolated_Electrons_px')
        py_electron = getattr(event, 'Isolated_Electrons_py')
        pz_electron = getattr(event, 'Isolated_Electrons_pz')
        
        for px, py, pz in zip(px_electron, py_electron, pz_electron):
            px_.append(px)
            py_.append(py)
            pz_.append(pz)
            energy_electron = np.sqrt(px**2 + py**2 + pz**2)
            energy_.append(energy_electron)

    def _add_muons(self, event, energy_, px_, py_, pz_):
        """Add muons to the 4-vector lists"""
        px_muons = getattr(event, 'Isolated_Muons_px')
        py_muons = getattr(event, 'Isolated_Muons_py')
        pz_muons = getattr(event, 'Isolated_Muons_pz')
        
        for px, py, pz in zip(px_muons, py_muons, pz_muons):
            px_.append(px)
            py_.append(py)
            pz_.append(pz)
            energy_muon = np.sqrt(px**2 + py**2 + pz**2)
            energy_.append(energy_muon)

    def _add_photons(self, event, energy_, px_, py_, pz_):
        """Add photons to the 4-vector lists"""
        px_photons = getattr(event, 'Isolated_Photons_px')
        py_photons = getattr(event, 'Isolated_Photons_py')
        pz_photons = getattr(event, 'Isolated_Photons_pz')
        
        for px, py, pz in zip(px_photons, py_photons, pz_photons):
            px_.append(px)
            py_.append(py)
            pz_.append(pz)
            energy_photon = np.sqrt(px**2 + py**2 + pz**2)
            energy_.append(energy_photon)

    def _add_tau_jets(self, event, energy_, px_, py_, pz_):
        """Add individual tau jets to the 4-vector lists"""
        tau_px = list(getattr(event, f'{self.jet_type}_px'))
        tau_py = list(getattr(event, f'{self.jet_type}_py'))
        tau_pz = list(getattr(event, f'{self.jet_type}_pz'))
        
        for px, py, pz in zip(tau_px, tau_py, tau_pz):
            energy = np.sqrt(px**2 + py**2 + pz**2)
            energy_.append(energy)
            px_.append(px)
            py_.append(py)
            pz_.append(pz)

    def __loadpN(self, energy_list, px_list, py_list, pz_list, number_to_pad_to):
        """Load 4-vectors into numpy array"""
        pN_evt = np.zeros((number_to_pad_to, 4))
        for i in range(number_to_pad_to):
            if i < len(energy_list):
                pN_evt[i][0] = energy_list[i]
                pN_evt[i][1] = px_list[i]
                pN_evt[i][2] = py_list[i]
                pN_evt[i][3] = pz_list[i]
            else:
                # Apply zero padding
                pN_evt[i][0] = 0
                pN_evt[i][1] = 0
                pN_evt[i][2] = 0
                pN_evt[i][3] = 0
        return pN_evt
