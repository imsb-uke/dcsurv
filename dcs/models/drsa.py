from tensorflow.keras.layers import Dropout, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.utils.validation import check_is_fitted
import tensorflow.keras.backend
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dcs.models.drsa_base import DrsaBase
import tensorflow as tf
import dcs.util


class Drsa(DrsaBase):

    def __init__(self,

                 #  architecture
                 encoder_num_layers=1,
                 encoder_nodes_per_layer=32,

                 decoder_num_layers=1,
                 decoder_nodes_per_layer=128,

                 # output dense layer
                 output_nodes_per_layer=32,

                 # output time output
                 output_grid_type='linear',
                 output_grid_num_nodes=5,
                 output_grid_max_quantile=1,

                 interpolate_method=None,

                 #  loss
                 alpha=0.25,

                 #  training
                 batch_size=128,
                 dropout_rate=0,
                 validation_size=0,
                 learning_rate=0.0001,
                 adam_clipvalue=None,
                 epochs=10,
                 use_early_stopping=False,
                 early_stopping_min_delta=0,
                 early_stopping_patience=0,
                 optimizer='Adam',

                 verbose=False,

                 ):

        self.output_grid_type = output_grid_type
        self.output_grid_num_nodes = output_grid_num_nodes
        self.output_grid_max_quantile = output_grid_max_quantile

        self.encoder_num_layers = encoder_num_layers
        self.encoder_nodes_per_layer = encoder_nodes_per_layer

        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.validation_size = validation_size
        self.learning_rate = learning_rate
        self.adam_clipvalue = adam_clipvalue
        self.epochs = epochs
        self.optimizer = optimizer

        self.use_early_stopping = use_early_stopping
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose

        self.output_nodes_per_layer = output_nodes_per_layer

        self.interpolate_method = interpolate_method

        self.decoder_num_layers = decoder_num_layers
        self.decoder_nodes_per_layer = decoder_nodes_per_layer

        # fixed:
        self.encoder_activation = 'relu'
        self.output_activation = 'sigmoid'
        self.output_num_layers = 1
        self.beta = 1
        self.loss_normalize = False
        self.loss_normalize_buckets = False
        self.time_input_type = 'time'
        self.time_input_normalize = False
        self.decoder_type = 'LSTM'
        self.decoder_activation = 'tanh'
        self.use_ipcw = False
        self.ipcw_max_factor = 10
        self.batch_size = batch_size

        super().__init__(
            output_grid_type=self.output_grid_type,
            output_grid_num_nodes=self.output_grid_num_nodes,
            output_grid_max_quantile=self.output_grid_max_quantile,

            encoder_num_layers=self.encoder_num_layers,
            encoder_nodes_per_layer=self.encoder_nodes_per_layer,
            encoder_activation=self.encoder_activation,

            output_num_layers=self.output_num_layers,
            output_nodes_per_layer=self.output_nodes_per_layer,

            output_activation=self.output_activation,

            dropout_rate=self.dropout_rate,

            alpha=self.alpha,
            beta=self.beta,
            loss_normalize=self.loss_normalize,
            loss_normalize_buckets=self.loss_normalize_buckets,
            use_ipcw=self.use_ipcw,
            ipcw_max_factor=self.ipcw_max_factor,

            validation_size=self.validation_size,

            learning_rate=self.learning_rate,
            adam_clipvalue=self.adam_clipvalue,
            epochs=self.epochs,

            use_early_stopping=self.use_early_stopping,
            early_stopping_min_delta=self.early_stopping_min_delta,
            early_stopping_patience=self.early_stopping_patience,

            verbose=self.verbose
        )

    def _create_model(self):

        tensorflow.keras.backend.clear_session()
        model = Sequential()

        # embedding
        self.encoder_num_nodes = self._validate_list(self.encoder_num_nodes)

        for i, curr_encoder_num_nodes in enumerate(self.encoder_num_nodes):
            model.add(Dense(
                curr_encoder_num_nodes,
                activation=self.encoder_activation,
                dtype='float32',
                name=f'encoder_{i}'
            ))
            if self.dropout_rate > 0:
                model.add(Dropout(rate=self.dropout_rate))

        # decoding

        # 'parse' decoder class
        decoder_classes = {'SimpleRNN': SimpleRNN, 'LSTM': LSTM, 'GRU': GRU}
        self.decoder_class = decoder_classes[self.decoder_type]

        # add decoding layers
        for i in range(self.decoder_num_layers):
            layer = self.decoder_class(
                units=self.decoder_nodes_per_layer,
                activation=self.decoder_activation,
                return_sequences=True,  # returns all outputs as output layer
                recurrent_dropout=self.dropout_rate,
                name=f'decoder_{i}'
            )
            model.add(layer)

        # Output layers are fully connected and returns 1 output per time step
        # in the last layer
        if self.output_num_nodes is None:
            self.output_num_nodes = 1

        self.output_num_nodes = self._validate_list(self.output_num_nodes)

        if self.output_num_nodes[-1] != 1:
            raise ValueError("Last layer should return only 1 node,"
                             f"not {self.output_num_nodes[-1]}")

        for i, curr_output_num_nodes in enumerate(self.output_num_nodes):
            model.add(Dense(
                units=curr_output_num_nodes,
                activation=self.output_activation,
                name=f'output_{i}',
            ))

        model.compile(self.optimizer, loss=self.loss_fn)

        return model

    def _add_time_input(self, X):
        """Adds individual output grid points to input
        """

        X_extended = tf.repeat(
            tf.reshape(X, (X.shape[0], 1, -1)),
            repeats=self.output_grid_num_nodes,
            axis=1)

        if self.time_input_type in ['None', None]:
            return X_extended

        if self.time_input_type == 'time':
            timesteps_extended = (
                self.output_grid
                .reshape(1, -1, 1)
                .repeat(X.shape[0], axis=0)
            )
            if self.time_input_normalize:
                timesteps_extended = timesteps_extended / \
                    np.max(timesteps_extended)

        elif self.time_input_type == 'index':  # only append index of timesteps
            timesteps_extended = (
                np.arange(0, len(self.output_grid)).reshape(1, -1, 1)
                .repeat(X.shape[0], axis=0)
            )

            if self.time_input_normalize:
                timesteps_extended = timesteps_extended / \
                    np.max(timesteps_extended)
        else:
            raise ValueError(
                "time_input_type should be in [None, 'time', 'index']")

        result = np.concatenate((X_extended, timesteps_extended), axis=2)
        return result

    def fit(self,
            X, y):
        """
        X: dimensions (samples, input_timesteps=1, features)
        y: dimensions (samples, (days, has_event))
        """
        super().fit(y)

        X = self._validate_X(X)
        y = self._validate_y(y)

        self._train_y = y.copy()

        # expand X in time dimension
        X_extended = self._add_time_input(X)

        # training
        self.history = self.model_.fit(
            X_extended, y,
            batch_size=self.batch_size,
            validation_split=self.validation_size,
            epochs=self.epochs,
            verbose=0,
            callbacks=self._callbacks,
        )

        if self.verbose:
            self.model_.summary()

        return self

    def predict(self,
                X,
                return_event_rates=False):

        check_is_fitted(self)

        X_was_dataframe = isinstance(X, pd.DataFrame)
        if X_was_dataframe:
            original_index = X.index

        X = self._validate_X(X)

        X_extended = self._add_time_input(X)

        event_rates = self.model_.predict(X_extended).reshape(X.shape[0], -1)
        event_rates = pd.DataFrame(event_rates, columns=self.output_grid)

        if return_event_rates:
            result = event_rates
        else:
            result = self._predict_survival_curves(event_rates)

            if self.interpolate_method is not None:
                result = dcs.util.interpolate_prediction(
                    result,
                    np.unique(self._train_y),
                    interpolate_method=self.interpolate_method,
                    interpolate_order=3)

        if X_was_dataframe:
            result.index = original_index

        return result
