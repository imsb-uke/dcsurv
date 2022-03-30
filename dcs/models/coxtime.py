import torchtuples as tt
import numpy as np
from sklearn.base import BaseEstimator
# import dcs.preprocessing
from pycox.models.cox_time import MLPVanillaCoxTime
import pycox.models
import pandas as pd
import dcs.util


class CoxTime(BaseEstimator):

    def __init__(self,
                 num_layers=3,
                 nodes_per_layer=32,
                 dropout=.2,
                 learning_rate=0.001,
                 val_size=0.1,
                 batch_size=128,
                 epochs=128,
                 use_early_stopping=False,
                 batch_norm=True,
                 optimizer="Adam",
                 verbose=False
                 ):
        """
        Args:
            dropout (float): dropout rate
            learning_rate (float): learning rate
            val_size (float): validation size for validation set used for early stopping
            batch_size (int): Size of input batch,
            epochs (int): Maximum number of epochs,
            use_early_stopping (bool): Use early stopping in training
            batch_norm (bool): Use batch normalization after each layer
            verbose (bool): print additional information
            optimizer (string): "Adam", "AdamW","AdamWR","RMSprop","SGD"

        Returns:
            CoxTime: Coxtime model with desired parameters
        """
        self.verbose = verbose

        # model parameters
        self.num_layers = num_layers
        self.nodes_per_layer = nodes_per_layer
        self.dropout = dropout
        self.val_size = val_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.batch_norm = batch_norm

        # training parameters
        self.use_early_stopping = use_early_stopping
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def prepare_input(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if np.isnan(X).any():
            raise ValueError('X cannot contain NaN')
        X = X.astype("float32")

        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        if y is not None:
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

        self.__dict__.pop('model', None)  # delete previous knowledge of model

        X, y = self.prepare_input(X, y)

        # stratify by occured events
        X_train, X_val, y_train, y_val = dcs.preprocessing.train_test_split(
            X, y, test_size=self.val_size, stratify=y[:, 1])

        self.label_transformer = pycox.models.CoxTime.label_transform()
        y_train = self.label_transformer.fit_transform(*dcs.util.unzip_labels(y_train))
        y_val = self.label_transformer.transform(*dcs.util.unzip_labels(y_val))
        val = tt.tuplefy(X_val, y_val)

        # initialize model
        self.model = pycox.models.CoxTime(
            MLPVanillaCoxTime(X.shape[1],
                              self._get_network_architecture(self.num_layers, self.nodes_per_layer),
                              self.batch_norm, self.dropout),
            self.get_optim(self.optimizer),
            labtrans=self.label_transformer)

        self.model.optimizer.set_lr(self.learning_rate)

        callbacks = [tt.callbacks.EarlyStopping()] if self.use_early_stopping else None

        self.log = self.model.fit(input=X_train,
                                  target=y_train,
                                  batch_size=self.batch_size,
                                  epochs=self.epochs,
                                  callbacks=callbacks,
                                  verbose=self.verbose,
                                  val_data=val.repeat(10).cat(),
                                  val_batch_size=self.batch_size
                                  )

        # after fitting, the baseline hazard can be computed
        self.model.compute_baseline_hazards()

    def predict(self, X):

        X_was_dataframe = isinstance(X, pd.DataFrame)
        if X_was_dataframe:
            original_index = X.index

        X, _ = self.prepare_input(X)

        result = self.model.predict_surv_df(X).T

        if X_was_dataframe:
            result.index = original_index

        return result

    def plot(self):
        return self.log.plot()
