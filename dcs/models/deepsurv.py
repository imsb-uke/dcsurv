import pycox.models
import torchtuples as tt
import numpy as np
from sklearn.base import BaseEstimator
import pandas as pd
import dcs


class DeepSurv(BaseEstimator):

    def __init__(self,
                 num_layers=3,
                 nodes_per_layer=32,
                 dropout=.2,
                 learning_rate=0.001,
                 val_size=0.1,
                 batch_size=128,
                 batch_norm=True,
                 use_early_stopping=False,
                 early_stopping_min_delta=0,
                 early_stopping_patience=10,
                 epochs=128,
                 verbose=False,
                 optimizer="Adam",
                 ):
        """
        Args:
            num_nodes (list(int)): list of ints for number of nodes in hidden units
            dropout (float): dropout rate
            learning_rate (float): learning rate
            val_size (float): validation size for validation set used for early stopping
            verbose (bool): print additional information
            optimizer (string): "Adam", "AdamW","AdamWR","RMSprop","SGD"

        Returns:
            DeepSurv: The return value. True for success, False otherwise
        """
        self.num_layers = num_layers
        self.nodes_per_layer = nodes_per_layer
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.use_early_stopping = use_early_stopping
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_patience = early_stopping_patience
        self.epochs = epochs
        self.verbose = verbose
        self.optimizer = optimizer

    def prepare_input(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if np.isnan(X).any():
            raise ValueError('X cannot contain NaN')

        X = X.astype("float32")

        if y is not None:
            if isinstance(y, pd.DataFrame):
                y = y.to_numpy()

            if np.isnan(y).any():
                raise ValueError('y cannot contain NaN')
            y = y.astype("float32")

        return X, y

    def get_optim(self, optim_name):
        optim_dict = {"SGD": tt.optim.SGD,
                      "Adam": tt.optim.Adam,
                      "AdamW": tt.optim.AdamWR,
                      "AdamWR": tt.optim.AdamWR,
                      "RMSprop": tt.optim.RMSprop
                      }
        return optim_dict[optim_name]

    @staticmethod
    def _get_network_architecture(num_layers, nodes_per_layer):
        return [nodes_per_layer] * num_layers

    def fit(self, X, y):

        X, y = self.prepare_input(X, y)

        X_train, X_val, y_train, y_val = dcs.preprocessing.train_test_split(
            X, y, test_size=self.val_size)

        y_train = dcs.util.unzip_labels(y_train)
        y_val = dcs.util.unzip_labels(y_val)

        callbacks = []
        if self.use_early_stopping:
            es_callback = tt.callbacks.EarlyStopping(
                min_delta=self.early_stopping_min_delta,
                patience=self.early_stopping_patience)
            callbacks.append(es_callback)

        net = tt.practical.MLPVanilla(
            in_features=X.shape[1],
            num_nodes=self._get_network_architecture(self.num_layers, self.nodes_per_layer),
            out_features=1,
            batch_norm=self.batch_norm,
            dropout=self.dropout,
            output_bias=True)

        self.model_ = pycox.models.CoxPH(net, self.get_optim(self.optimizer))
        self.model_.optimizer.set_lr(self.learning_rate)

        self.log = self.model_.fit(X_train, y_train,
                                   self.batch_size,
                                   self.epochs,
                                   callbacks=callbacks,
                                   verbose=self.verbose,
                                   val_data=(X_val, y_val),
                                   val_batch_size=self.batch_size
                                   )

        # after fitting, the baseline hazard can be computed
        self.model_.compute_baseline_hazards()

    def predict(self, X):

        X_was_dataframe = isinstance(X, pd.DataFrame)
        if X_was_dataframe:
            original_index = X.index

        X, _ = self.prepare_input(X)

        result = self.model_.predict_surv_df(X).T

        if X_was_dataframe:
            result.index = original_index

        return result

    def plot_learning_curve(self):
        return self.log.plot()
