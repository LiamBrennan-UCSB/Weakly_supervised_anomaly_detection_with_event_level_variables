'''
Taken from: https://github.com/uhh-pd-ml/sk_cathode/blob/main/sk_cathode/utils/ensembling_utils.py
'''
import copy
import numpy as np
import os
from os.path import exists
from sklearn.base import BaseEstimator
import pickle

class EnsembleModel(BaseEstimator):
    """Wrapper class to combine multiple models into a single one.
    The models need to share the API that should be accessed later.
    E.g. ensemble_model.fit() will make the same call to the base models,
    while ensemble_model.predict() will call the predict() method of each
    base model and aggregate the results via the aggregation_func (default is
    np.mean).

    Parameters
    ----------
    base_model_list : list
        List of models to be ensembled. They need to have the same API.
    aggregation_func : function, default=np.mean
        Function to aggregate predictions of base models.
    avoid_overwriting : bool, default=False
        If True, existing models will not be overwritten during training.
    """

    def __init__(self, base_model_list, aggregation_func=np.mean,
                 avoid_overwriting=False):
        self.base_model_list = base_model_list
        self.aggregation_func = aggregation_func
        self.avoid_overwriting = avoid_overwriting

    def fit(self, *args, **kwargs):
        for i, model in enumerate(self.base_model_list):
            if self.avoid_overwriting:
                try:
                    does_exist = exists(model._model_path(0))
                except AttributeError:
                    does_exist = False
                if does_exist:
                    print(f"Model {i+1}/{len(self.base_model_list)} "
                          f"already exists. Skipping training.")
                    model.load_best_model()
                    continue

            print(f"Training model {i+1}/{len(self.base_model_list)}")
            model.fit(*args, **kwargs)

    def load_best_model(self, *args, **kwargs):
        for model in self.base_model_list:
            model.load_best_model(*args, **kwargs)

    def predict(self, *args, **kwargs):
        preds = []
        for model in self.base_model_list:
            preds.append(model.predict(*args, **kwargs))
        return self.aggregation_func(np.stack(preds, axis=0), axis=0)

    def predict_proba(self, *args, **kwargs):
        preds = []
        for model in self.base_model_list:
            preds.append(model.predict_proba(*args, **kwargs))
        return self.aggregation_func(np.stack(preds, axis=0), axis=0)

    def transform(self, *args, **kwargs):
        preds = []
        for model in self.base_model_list:
            preds.append(model.transform(*args, **kwargs))
        return self.aggregation_func(np.stack(preds, axis=0), axis=0)

    def sample(self, n_samples=1, **kwargs):
        raise NotImplementedError

    def predict_log_proba(self, *args, **kwargs):
        return np.log(self.predict_proba(*args, **kwargs))

    def score_samples(self, *args, **kwargs):
        raise NotImplementedError

    def score(self, *args, **kwargs):
        raise NotImplementedError


class EpochEnsembleModel(EnsembleModel):
    """Wrapper class to build an ensemble from multiple epochs of the
    same training.

    Parameters
    ----------
    base_model : BaseEstimator
        Model to be ensembled.
    n_best_epochs : int, default=10
        Number of best epochs to be ensembled.
    """

    def __init__(self, base_model, n_best_epochs=10):
        self.base_model = base_model
        self.n_best_epochs = n_best_epochs
        self.base_model_list = [base_model]

    def fit(self, *args, **kwargs):
        # only train base model
        self.base_model.fit(*args, **kwargs)

    def load_best_model(self):
        val_loss = self.base_model.load_val_loss()
        best_epochs = np.argsort(val_loss)[:self.n_best_epochs]
        self.base_model_list = [
            copy.deepcopy(self.base_model)
            for _ in range(self.n_best_epochs)]
        for i, epoch in enumerate(best_epochs):
            self.base_model_list[i].load_epoch_model(epoch)

# Add to utils/ensembling_utils.py