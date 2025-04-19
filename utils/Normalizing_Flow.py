"""
Conditional Normalizing Flow implementation based on nFlows, with scikit-learn-like API.
"""
import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.base import CompositeTransform
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split


class ConditionalNormalizingFlow(BaseEstimator):
    """
    Conditional normalizing flow based on nFlows with scikit-learn-like API.
    
    Parameters:
    -----------
    save_path : str, optional
        Path to save the model to. If None, no model is saved.
    load : bool, default=False
        Whether to load the model from save_path.
    num_inputs : int, default=4
        Number of inputs to the model being modeled.
    num_cond_inputs : int, default=1
        Number of conditional inputs to the model.
    num_blocks : int, default=10
        Number of transform blocks in the model.
    num_hidden : int, default=64
        Number of hidden units in each transform block.
    num_layers : int, default=2
        Number of layers in each transform block.
    num_bins : int, default=8
        Number of bins in the piecewise rational quadratic spline.
    tail_bound : int, default=10
        Bound for the tails of the spline.
    batch_norm : bool, default=False
        Whether to use batch normalization.
    lr : float, default=0.0001
        Learning rate for the optimizer.
    weight_decay : float, default=0.000001
        Weight decay for the optimizer.
    early_stopping : bool, default=False
        Whether to use early stopping.
    patience : int, default=10
        Number of epochs to wait for improvement before stopping.
    no_gpu : bool, default=False
        Whether to disable GPU usage.
    batch_size : int, default=256
        Batch size during training.
    drop_last : bool, default=True
        Whether to drop the last batch if it's smaller than batch_size.
    epochs : int, default=100
        Maximum number of epochs to train for.
    verbose : bool, default=False
        Whether to print progress during training.
    run_name : str, optional
        Name for the current run, used for saving files.
    """
    def __init__(self, save_path=None, load=False,
                 num_inputs=4, num_cond_inputs=1, num_blocks=10,
                 num_hidden=64, num_layers=2, num_bins=8,
                 tail_bound=10, batch_norm=False, lr=0.0001,
                 weight_decay=0.000001, early_stopping=False,
                 patience=10, no_gpu=False, batch_size=256,
                 drop_last=True, epochs=100, verbose=False,
                 run_name=None):

        self.save_path = save_path
        if save_path is not None:
            self.de_model_path = os.path.join(save_path, "DE_models/")
        else:
            self.de_model_path = None
        self.load = load

        self.no_gpu = no_gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not no_gpu else "cpu")
        self.early_stopping = early_stopping
        self.patience = patience
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.epochs = epochs
        self.verbose = verbose

        self.num_inputs = num_inputs
        self.num_cond_inputs = num_cond_inputs
        self.num_blocks = num_blocks
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.batch_norm = batch_norm
        self.lr = lr
        self.weight_decay = weight_decay
        
        # File management parameters
        self.run_name = run_name
        self.num_cycles = num_blocks

        # Build the flow model
        self._build_model()
        
        # Initialize the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Set to evaluation mode by default
        self.model.eval()

        if load:
            self.load_best_model()
            
    def _build_model(self):
        """Builds the normalizing flow model."""
        # Create the base distribution
        base_dist = StandardNormal(shape=[self.num_inputs])

        # Create the transform modules
        modules = []
        for i in range(self.num_cycles):
            # Add autoregressive transform
            modules.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    num_bins=self.num_bins,
                    tails='linear',
                    tail_bound=self.tail_bound,
                    features=self.num_inputs,
                    context_features=self.num_cond_inputs,
                    hidden_features=self.num_hidden,
                    num_blocks=self.num_blocks,
                    random_mask=False,
                    activation=nn.ReLU(),
                    dropout_probability=0.,
                    use_batch_norm=self.batch_norm,
                    use_residual_blocks=False
                )
            )
            # Add permutation
            modules.append(ReversePermutation(features=self.num_inputs))

        # Create the composite transform
        transform = CompositeTransform(modules)
        
        # Create the flow model
        self.model = Flow(transform, base_dist)
        self.model.to(self.device)
        
        # Log model size
        total_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"ConditionalNormalizingFlow has {total_parameters} parameters")

    def fit_noPenalty(self, X, m, X_val=None, m_val=None):
        """
        Fits the model to the provided data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray
            Conditional data.
        X_val : numpy.ndarray, optional
            Validation input data.
        m_val : numpy.ndarray, optional
            Validation conditional data.
            
        Returns:
        --------
        self : object
            Fitted estimator.
        """
        # Validate inputs
        assert not (self.epochs is None and not self.early_stopping), (
            "A finite number of epochs must be set if early stopping is not used!")

        if self.drop_last:
            assert X.shape[0] >= self.batch_size, (
                "If drop_last is True, X needs to have at least as many samples as the batch size!")

        # If validation data not provided, create from training data
        if X_val is None and m_val is None:
            X_train, X_val, m_train, m_val = train_test_split(X, m, test_size=0.2, shuffle=True)
        else:
            X_train = X.copy()
            m_train = m.copy()

        # Create model save directory if needed
        if self.de_model_path is not None:
            os.makedirs(self.de_model_path, exist_ok=True)

        # Remove NaN values
        nan_mask = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[nan_mask]
        m_train = m_train[nan_mask]

        nan_mask = ~np.isnan(X_val).any(axis=1)
        X_val = X_val[nan_mask]
        m_val = m_val[nan_mask]

        # Create data loaders
        train_loader = self._numpy_to_torch_loader(X_train, m_train)
        val_loader = self._numpy_to_torch_loader(X_val, m_val, shuffle=False, drop_last=False)

        # Set model to training mode
        self.model.train()
        
        # Calculate initial losses
        train_loss = self._compute_loss(train_loader)
        val_loss = self._compute_loss(val_loader)
        
        train_losses = np.array([train_loss])
        val_losses = np.array([val_loss])
        
        # Save initial losses
        if self.save_path is not None:
            np.save(self._train_loss_path(), train_losses)
            np.save(self._val_loss_path(), val_losses)
        
        # Training loop
        best_val_loss = val_loss
        best_epoch = 0
        epochs_without_improvement = 0
        
        for epoch in range(self.epochs if self.epochs is not None else 1000):
            print(f'\nEpoch: {epoch}')
            
            # Train for one epoch
            train_loss = self._train_epoch(train_loader)
            
            # Evaluate on validation set
            val_loss = self._compute_loss(val_loader)
            
            # Handle NaN loss
            if np.isnan(val_loss):
                raise ValueError("Training yields NaN validation loss!")
            
            # Print progress
            print(f"Train loss: {train_loss:.6f}")
            print(f"Validation loss: {val_loss:.6f}")
            
            # Save losses
            train_losses = np.append(train_losses, train_loss)
            val_losses = np.append(val_losses, val_loss)
            
            if self.save_path is not None:
                np.save(self._train_loss_path(), train_losses)
                np.save(self._val_loss_path(), val_losses)
                self._save_model(self._model_path(epoch))
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                # Save best model
                if self.save_path is not None:
                    self._save_model(self._best_model_path())
            else:
                epochs_without_improvement += 1
            
            print(f"Epochs without improvement: {epochs_without_improvement}")
            if self.early_stopping and epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch}: No improvement since epoch {best_epoch}")
                break
        
        # Set model back to evaluation mode
        self.model.eval()
        
        # Save final losses
        if self.save_path is not None:
            np.save(self._train_loss_path(), train_losses)
            np.save(self._val_loss_path(), val_losses)
            
            # Load best model
            self.load_best_model()
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        
        return self
    
    def _train_epoch(self, train_loader):
        """
        Train the model for one epoch.
        
        Parameters:
        -----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data.
            
        Returns:
        --------
        float : Average training loss for the epoch.
        """
        self.model.train()
        train_loss = 0
        train_loss_vals = []
        
        # Progress bar if verbose
        iterator = tqdm(train_loader) if self.verbose else train_loader
        
        # Process each batch
        for batch_idx, (data, cond_data) in enumerate(iterator):
            data = data.to(self.device)
            cond_data = cond_data.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute negative log-likelihood
            loss = -self.model.log_prob(data, cond_data)
            
            # Average loss over batch
            batch_loss = loss.mean()
            train_loss += batch_loss.item()
            train_loss_vals.extend(loss.tolist())
            
            # Backpropagation
            batch_loss.backward()
            self.optimizer.step()
            
            # Update progress bar
            if self.verbose:
                iterator.set_description(
                    f"Train, Log likelihood in nats: {-train_loss / (batch_idx + 1):.6f}"
                )
        
        # Return average loss
        return np.array(train_loss_vals).mean()
    
    def _compute_loss(self, data_loader):
        """
        Compute loss over a dataset.
        
        Parameters:
        -----------
        data_loader : torch.utils.data.DataLoader
            DataLoader for evaluation data.
            
        Returns:
        --------
        float : Average loss over the dataset.
        """
        self.model.eval()
        total_loss = 0
        n_batches = 0
        n_nans = 0
        n_highs = 0
        
        with torch.no_grad():
            for batch_idx, (data, cond_data) in enumerate(data_loader):
                data = data.to(self.device)
                cond_data = cond_data.to(self.device)
                
                # Compute log probabilities
                log_probs = self.model.log_prob(data, cond_data)
                log_probs = log_probs.flatten()
                
                # Count problematic values
                n_nans += torch.isnan(log_probs).sum().item()
                n_highs += (torch.abs(log_probs) >= 1000).sum().item()
                
                # Filter out problematic values
                valid_log_probs = log_probs[~torch.isnan(log_probs) & (torch.abs(log_probs) < 1000)]
                
                if len(valid_log_probs) > 0:
                    loss = -valid_log_probs.mean().item()
                    total_loss += loss
                    n_batches += 1
        
        # Report problematic values
        print(f"NaNs: {n_nans}, High values: {n_highs}")
        
        # Return average loss
        return total_loss / max(1, n_batches)
    
    def _numpy_to_torch_loader(self, X, m, shuffle=True, drop_last=None):
        """
        Convert numpy arrays to PyTorch DataLoader.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray
            Conditional data.
        shuffle : bool, default=True
            Whether to shuffle the data.
        drop_last : bool, optional
            Whether to drop the last batch if it's smaller than batch_size.
            
        Returns:
        --------
        torch.utils.data.DataLoader : DataLoader for the data.
        """
        if drop_last is None:
            drop_last = self.drop_last
            
        # Convert to PyTorch tensors
        X_torch = torch.from_numpy(X).float().to(self.device)
        m_torch = torch.from_numpy(m).float().to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_torch, m_torch)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            drop_last=drop_last
        )
        
        return dataloader
    
    def transform(self, X, m=None):
        """
        Transform data to latent space.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray
            Conditional data.
            
        Returns:
        --------
        numpy.ndarray : Latent space representation.
        """
        if m is None:
            raise ValueError("Conditional data m must be provided")
            
        with torch.no_grad():
            X_torch = torch.from_numpy(X).float().to(self.device)
            m_torch = torch.from_numpy(m).float().to(self.device)
            Xt, _ = self.model._transform.forward(X_torch, context=m_torch)
            
        return Xt.cpu().numpy()
    
    def inverse_transform(self, Xt, m=None):
        """
        Transform data from latent space to original space.
        
        Parameters:
        -----------
        Xt : numpy.ndarray
            Latent space data.
        m : numpy.ndarray
            Conditional data.
            
        Returns:
        --------
        numpy.ndarray : Original space representation.
        """
        if m is None:
            raise ValueError("Conditional data m must be provided")
            
        with torch.no_grad():
            Xt_torch = torch.from_numpy(Xt).float().to(self.device)
            m_torch = torch.from_numpy(m).float().to(self.device)
            X, _ = self.model._transform.inverse(Xt_torch, context=m_torch)
            
        return X.cpu().numpy()
    
    def sample(self, n_samples=1, m=None):
        """
        Sample from the model.
        
        Parameters:
        -----------
        n_samples : int, default=1
            Number of samples to draw.
        m : numpy.ndarray
            Conditional data.
            
        Returns:
        --------
        numpy.ndarray : Samples from the model.
        """
        if m is None:
            raise ValueError("Conditional data m must be provided")
            
        with torch.no_grad():
            m_torch = torch.from_numpy(m).float().to(self.device)
            X = self.model.sample(n_samples, context=m_torch).reshape(n_samples, -1)
            
        return X.cpu().numpy()
    
    def predict_log_proba(self, X, m=None):
        """
        Predict log probability of the data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray
            Conditional data.
            
        Returns:
        --------
        numpy.ndarray : Log probability of the data.
        """
        if m is None:
            raise ValueError("Conditional data m must be provided")
            
        with torch.no_grad():
            X_torch = torch.from_numpy(X).float().to(self.device)
            m_torch = torch.from_numpy(m).float().to(self.device)
            log_prob = self.model.log_prob(X_torch, m_torch)
            
        return log_prob.cpu().numpy().reshape(-1, 1)
    
    def predict_proba(self, X, m=None):
        """
        Predict probability of the data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray
            Conditional data.
            
        Returns:
        --------
        numpy.ndarray : Probability of the data.
        """
        return np.exp(self.predict_log_proba(X, m))
    
    def load_best_model(self):
        """Load the best model."""
        print("Loading best model...")
        
        # Check if best model exists
        best_model_path = self._best_model_path()
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print(f"Loaded model from {best_model_path}")
        else:
            print(f"Best model file not found at {best_model_path}")
            
            # Try to find the latest epoch model
            if self.de_model_path is not None and os.path.exists(self.de_model_path):
                model_files = [f for f in os.listdir(self.de_model_path) if f.endswith('.par')]
                if model_files:
                    latest_model = sorted(model_files)[-1]
                    latest_path = os.path.join(self.de_model_path, latest_model)
                    self.model.load_state_dict(torch.load(latest_path, map_location=self.device))
                    print(f"Loaded latest model from {latest_path}")
        
        # Set to evaluation mode
        self.model.eval()
    
    def load_epoch_model(self, epoch):
        """
        Load model state from a specific epoch.
        
        Parameters:
        -----------
        epoch : int
            Epoch to load model from.
        """
        model_path = self._model_path(epoch)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from epoch {epoch}")
        else:
            print(f"Model file not found at {model_path}")
    
    def _save_model(self, path):
        """
        Save model state to a file.
        
        Parameters:
        -----------
        path : str
            Path to save model to.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def _train_loss_path(self):
        """Get path for training loss file."""
        return os.path.join(self.save_path, f"{self.run_name}_DE_train_losses.npy")
    
    def _val_loss_path(self):
        """Get path for validation loss file."""
        return os.path.join(self.save_path, f"{self.run_name}_DE_val_losses.npy")
    
    def _model_path(self, epoch):
        """Get path for model file at a specific epoch."""
        return os.path.join(self.de_model_path, f"{self.run_name}_DE_epoch_{epoch}.par")
        
    def _best_model_path(self):
        """Get path for best model file."""
        return os.path.join(self.de_model_path, f"{self.run_name}_DE_best_model.par")