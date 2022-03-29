import numpy as np
import pandas as pd

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Bidirectional, \
    RepeatVector, Reshape, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import extend_with_decoupled_weight_decay
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.models import Model, Sequential
from tensorflow import convert_to_tensor

from tqdm.keras import TqdmCallback
from sklearn.utils.validation import check_is_fitted

import matplotlib.pyplot as plt

from dcs import defaults
from dcs.util import interpolate_prediction, hazard_to_survival
from dcs.models.dcs_loss import DcsLoss


class DcsModel(KerasRegressor):

    def __init__(self, build_fn=None, **sk_params):
        self.sk_params = sk_params
        super().__init__(build_fn=None, **sk_params)

    _valid_activations = ['relu', 'sigmoid', 'tanh']
    _valid_output_grid_types = ['linear', 'quantile', 'log']

    def _validate_params(self):

        greater_0_params = {
            "learning_rate": self.learning_rate,
            "sigma": self.sigma,
            "output_num_layers": self.output_num_layers,
            "output_grid_num_nodes": self.output_grid_num_nodes,
            "output_grid_max": self.output_grid_max,
        }
        for param_name, param in greater_0_params.items():
            if param <= 0:
                raise ValueError(f"{param_name} should be > 0.")

        geq_0_params = {
            "lambda_": self.lambda_,
        }
        for param_name, param in geq_0_params.items():
            if param < 0:
                raise ValueError(f"{param_name} should be >= 0.")

        # activations
        if self.output_activation.lower() not in self._valid_activations:
            raise ValueError(f"output_activation should be one of {self._valid_activations}")

        if self.output_grid_type not in self._valid_output_grid_types:
            raise ValueError(f"output_grid_type should be one of {self._valid_output_grid_types}")

    def __call__(self,
                 X, y,
                 output_grid_type='linear', output_grid_num_nodes=5, output_grid_max=None,
                 output_as_hazard=True,
                 output_num_layers=1, output_nodes_per_layer=32, output_activation='sigmoid',
                 encoder_num_layers=1, encoder_nodes_per_layer=32, encoder_activation='relu',
                 decoder_num_layers=1, decoder_nodes_per_layer=32,
                 decoder_activation='tanh', decoder_recurrent_activation='sigmoid',
                 decoder_bidirectional=False, decoder_use_lstm_skip=False,
                 dropout_rate=0,
                 use_early_stopping=False, early_stopping_patience=0, early_stopping_min_delta=0,
                 validation_size=0,
                 interpolate_method='pad',
                 lambda_=1, sigma=1,
                 loss_normalize=True,
                 loss_kernel_include_censored=True,
                 optimizer='Adam',
                 adam_clipvalue=None,
                 adamw_weight_decay=0.0001,
                 epochs=1, learning_rate=.001, run_eagerly=False,
                 verbose=False):

        # region save parameters as attributes

        # # output_grid
        self.output_grid_type = output_grid_type
        self.output_grid_num_nodes = output_grid_num_nodes
        self.output_grid_max = output_grid_max
        self._set_output_grid(y[defaults.DURATION_COL])

        # # network
        self.encoder_num_layers = encoder_num_layers
        self.encoder_nodes_per_layer = encoder_nodes_per_layer
        self.encoder_activation = encoder_activation

        self.decoder_num_layers = decoder_num_layers
        self.decoder_activation = decoder_activation
        self.decoder_recurrent_activation = decoder_recurrent_activation
        self.decoder_use_lstm_skip = decoder_use_lstm_skip
        self.decoder_bidirectional = decoder_bidirectional
        self.decoder_use_lstm_skip = decoder_use_lstm_skip

        self.decoder_nodes_per_layer = decoder_nodes_per_layer

        self.output_num_layers = output_num_layers
        self.output_nodes_per_layer = output_nodes_per_layer
        self.output_activation = output_activation

        self.interpolate_method = interpolate_method

        self.dropout_rate = dropout_rate

        # # loss
        self.output_as_hazard = output_as_hazard
        self.lambda_ = lambda_
        self.sigma = sigma
        self.loss_normalize = loss_normalize
        self.loss_kernel_include_censored = loss_kernel_include_censored

        # # training
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.run_eagerly = run_eagerly

        self.optimizer = optimizer
        self.adam_clipvalue = adam_clipvalue
        self.adamw_weight_decay = adamw_weight_decay

        self.validation_size = validation_size

        self.use_early_stopping = use_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        self.verbose = verbose

        # endregion

        # region build model
        input_ = Input(shape=X.shape[1])

        model_builder = input_

        # encoder layer(s)
        for i in range(self.encoder_num_layers):
            model_builder = Dense(
                self.encoder_nodes_per_layer,
                activation=self.encoder_activation,
                dtype='float32',
                name=f'encoder_{i}',
            )(model_builder)
            if self.dropout_rate > 0:
                model_builder = Dropout(rate=self.dropout_rate)(model_builder)

        # decoder layer(s)
        if self.decoder_num_layers > 0:
            model_builder = RepeatVector(self.output_grid_num_nodes)(model_builder)

            pre_lstm = Lambda(lambda x: x, name='Identity')(model_builder)

            for i in range(self.decoder_num_layers):
                lstm_class = LSTM(units=self.decoder_nodes_per_layer,
                                  return_sequences=True,
                                  activation=self.decoder_activation,
                                  recurrent_activation=self.decoder_recurrent_activation,
                                  recurrent_dropout=self.dropout_rate,
                                  name=f'decoder_{i}')

                if self.decoder_bidirectional:
                    lstm_class = Bidirectional(lstm_class)

                model_builder = lstm_class(model_builder)

            if self.decoder_use_lstm_skip:
                model_builder = Concatenate(axis=2)([model_builder, pre_lstm])

        # output layer(s)
        for i in range(self.output_num_layers - 1):
            model_builder = Dense(
                self.output_nodes_per_layer,
                activation=self.output_activation,
                dtype='float32',
                name=f'output_{i}',
            )(model_builder)
            if self.dropout_rate > 0:
                model_builder = Dropout(rate=self.dropout_rate)(model_builder)

        if self.decoder_num_layers > 0:
            model_builder = Dense(1, activation=self.output_activation)(model_builder)
            model_builder = Reshape([-1])(model_builder)
        else:
            model_builder = Dense(self.output_grid_num_nodes,
                                  activation=self.output_activation)(model_builder)

        model = Model(inputs=input_,
                      outputs=model_builder)

        opt = Adam(
            learning_rate=self.learning_rate,
            clipvalue=self.adam_clipvalue,
        )

        if self.optimizer.lower() == 'adamw':

            opt = extend_with_decoupled_weight_decay(Adam)(
                self.adamw_weight_decay,
                learning_rate=self.learning_rate,
                clipvalue=self.adam_clipvalue)

        model.compile(
            loss=DcsLoss(
                self._output_grid,
                normalize=self.loss_normalize,
                lambda_=self.lambda_,
                sigma=self.sigma,
                y_pred_is_hazard=self.output_as_hazard,
                kernel_include_censored=self.loss_kernel_include_censored).loss_fn,
            optimizer=opt,
            run_eagerly=self.run_eagerly,
        )

        if self.verbose:
            print(model.summary())

        # end region

        return model

    def _set_output_grid(self, durations):

        if self.output_grid_max in [None, 'None']:
            self.output_grid_max = durations.max() + 1

        if self.output_grid_type == 'linear':
            self._output_grid = np.linspace(
                0,
                self.output_grid_max,
                self.output_grid_num_nodes + 1)

        elif self.output_grid_type == 'log':
            self._output_grid = [0, *np.logspace(
                1, np.log10(self.output_grid_max),
                self.output_grid_num_nodes + 1)[1:]]

        elif self.output_grid_type == 'quantile':
            self._output_grid = [0, *durations.quantile(
                np.linspace(0, 1, self.output_grid_num_nodes + 1)).values[1:]]
        else:
            raise NotImplementedError(f"'{self.output_grid_type} not implemented!'")

        self._output_grid = np.round(self._output_grid, 0)

    def plot_history(self):
        if not hasattr(self, 'history'):
            raise ValueError('No history to plot! Call fit first.')

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(self.history['loss'], label='loss')
        ax.plot(self.history['val_loss'], label='val_loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.show()

    def fit(self, X, y, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.
        Args:
            x : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`
        Returns:
            history : object
                details about the training history at each epoch.
        """

        self._fit_y = y
        self._fit_X = X

        self.model = self.__call__(X=X, y=y, **self.filter_sk_params(self.__call__))

        self._validate_params()

        # Callbacks
        self._callbacks = [TerminateOnNaN()]

        if self.verbose:
            self._callbacks.append(TqdmCallback(verbose=1))

        if self.use_early_stopping:

            if self.validation_size == 0:
                raise ValueError("Use of early stopping requires validation_size > 0")

            self._callbacks.append(
                EarlyStopping(monitor='val_loss',
                              mode='min',
                              verbose=self.verbose,
                              min_delta=self.early_stopping_min_delta,
                              patience=self.early_stopping_patience,
                              restore_best_weights=True
                              ))

        import copy
        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
        fit_args.update(kwargs)
        fit_args.pop('verbose', None)
        self.history = self.model.fit(
            X, y,
            validation_split=self.validation_size,
            callbacks=self._callbacks,
            verbose=False,
            **fit_args).history

        return self

    def predict(self, X):

        check_is_fitted(self)

        result = super().predict(X)

        if self.output_as_hazard:
            result = hazard_to_survival(convert_to_tensor(result)).numpy()

        result = pd.DataFrame(result, index=X.index, columns=self._output_grid[1:])

        if self.interpolate_method is not None:
            result = interpolate_prediction(
                result,
                np.unique(self._fit_y),
                interpolate_method=self.interpolate_method,
                interpolate_order=3)

        return result
