from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
import warnings
from tqdm.keras import TqdmCallback
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, Callback
import os
from lifelines import KaplanMeierFitter

from dcs import defaults


class DrsaBase(BaseEstimator):

    def __init__(self,

                 output_grid_type='linear',
                 output_grid_num_nodes=5,
                 output_grid_max_quantile=1,

                 encoder_num_layers=1,
                 encoder_nodes_per_layer=32,
                 encoder_activation='sigmoid',

                 output_num_layers=1,
                 output_nodes_per_layer=1,

                 output_activation='sigmoid',

                 dropout_rate=0,

                 alpha=0.25,
                 beta=1,
                 loss_normalize=False,
                 loss_normalize_buckets=False,
                 use_ipcw=False,
                 ipcw_max_factor=10,

                 validation_size=0,

                 learning_rate=0.0001,
                 adam_clipvalue=None,
                 epochs=10,

                 use_early_stopping=False,
                 early_stopping_min_delta=0,
                 early_stopping_patience=0,

                 verbose=False):
        """Drsa Base model for survival prediction

        Based on [1]

        Args:
        -----
            output_grid_type (string): 'linear', 'log' or 'quantile'
            output_grid_num_nodes (int): number of output nodes

            alpha (float): relative loss weight - alpha * L_z + (1-alpha) * L_c
            beta (float): relative loss weight - L_cen + beta * L_unc
            loss_normalize (bool): normalize loss to no. of batch samples?
            loss_normalize_buckets (bool): normalize loss by number of buckets?

            encoder_num_nodes (int or list): set encoding NN structure.
                Either as int (64) or list ([16, 32, 16]) for stacking
            encoder_activation (string): encoder activation function
            output_num_nodes (list): override one layer output dense layer with this NN structure
            output_activation (string): output layer activation function

            dropout_rate (float): dropout in encoder, decoder and output layers

            learning_rate (float): learning rate
            adam_clipvalue (float): clip gradients in adam optimizer
            validation_size (float): size of validation split during training
            epochs (int): max epochs to train

            use_early_stopping (bool): use early stopping?
            early_stopping_min_delta (float): minimum change considered for
                a better model
            early_stopping_patience (int): no. of epochs without improvement
                before stopping

            verbose (bool): print additional information

        References
        ----
        [1] Ren et al 2018: Deep Recurrent Survival Analysis
        """

        # Output Grid
        self.output_grid_type = output_grid_type
        self.output_grid_num_nodes = output_grid_num_nodes
        self.output_grid_max_quantile = output_grid_max_quantile

        # Architecture
        self.encoder_activation = encoder_activation
        self.encoder_num_nodes = self._convert_num_nodes(encoder_num_layers,
                                                         encoder_nodes_per_layer)
        self.output_activation = output_activation
        self.output_num_nodes = self._convert_num_nodes(output_num_layers,
                                                        output_nodes_per_layer)

        if self.output_num_nodes != [1]:
            self.output_num_nodes = [*self.output_num_nodes, 1]  # last output layer forced 1

        # Learning
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.adam_clipvalue = adam_clipvalue
        self.validation_size = validation_size
        self.dropout_rate = dropout_rate

        # loss
        self.alpha = alpha
        self.beta = beta
        self.loss_normalize = loss_normalize
        self.loss_normalize_buckets = loss_normalize_buckets
        self.use_ipcw = use_ipcw
        self.ipcw_max_factor = ipcw_max_factor

        self.verbose = verbose

        # Early Stopping
        self.use_early_stopping = use_early_stopping
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_patience = early_stopping_patience

        # set additional variables
        self._develop_mode = True  # TODO remove?
        self.title = "BASE"  # overwrite this in child models

    @staticmethod
    def _convert_num_nodes(num_layers, nodes_per_layer):
        return [nodes_per_layer] * num_layers

    def __get_output_intervals(self):

        output_intervals = []
        for i, t in enumerate(self.output_grid):
            if i == 0:
                continue
            if i == 1:
                interval = f"(0,{t}]"
            if i == len(self.output_grid) - 1:
                interval = f"({t}, inf)"
            else:
                interval = f"({self.output_grid[i-1]}, {self.output_grid[i]}]"

            output_intervals.append(interval)

        return output_intervals
    _output_intervals = property(__get_output_intervals)

    def plot_learning_curve(self, ax=None):

        check_is_fitted(self)

        if ax is None:
            ax = plt.gca()

        history = self.history.history

        metrics = ['loss', 'val_loss']
        for metric in metrics:
            if metric in history.keys():
                ax.plot(history[metric],
                        label=metric,
                        markersize=2,)

        ax.set_title(self.title + ' model loss')
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.legend()

    def _interpolate_df(self,
                        df,
                        target_grid,
                        interpolate_method='pad',
                        interpolate_order=3):
        """
        Interpolate df assuming columns are days

        interpolate_method can be 'pad', 'linear', 'quadratic', 'cubic', spline'
        """
        df_interp = (
            df
            .assign(at_0=1)
            .assign(at_1=lambda df: df.iloc[:, 0]
                    if interpolate_method == 'pad' else np.nan)
            .rename(columns={'at_0': 0, 'at_1': 1})
            .T
            .assign(index=lambda df:
                    pd.to_timedelta(df.index, unit='days')).set_index('index')

            .resample(rule=pd.Timedelta(days=1)).mean()

            .interpolate(method=interpolate_method,
                         order=interpolate_order)
            .assign(index=lambda df: df.index.days).set_index('index')

            .rename_axis('', axis=0)
            .T
        )

        # choose all of target grid that are present in interpolated grid
        final_grid = np.sort(np.array(
            list(set(df_interp.columns).intersection(set(target_grid))),
        ))

        final_grid = [0, *final_grid]

        return df_interp.loc[:, final_grid]

    def predict(self,
                X,
                return_event_rates=False):
        warnings.warn("Implement this method in inheritance.")
        pass

    def predict_plot(self,
                     X,
                     Y=None,
                     include_hazards=False,
                     xscale='linear',
                     hazards_ymax=None,
                     ax=None,
                     ):
        """Plots survival curve for input X

        Args:
        ----
            X (df or numpy): Input individuals dims (n_patients, n_features)
            ax: plot to this axis object
        """

        survival_curves = self.predict(X)

        if ax is None:
            ax = plt.gca()

        # Survival curve is between 0 and 1
        ymin, ymax = 0, 1.1
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("$S(t|x_i)$")
        ax.set_xlabel("days")
        ax.set_xscale(xscale)

        if include_hazards:
            hazards = self.predict(X, return_event_rates=True)
            ax2 = ax.twinx()
            ax2.set_ylabel("$h(t|x_i)$")
            ax2.set_ylim(0, hazards_ymax)

        # plot bucket regions in alternating background colors
        for i in range(1, len(self.output_grid)):
            # fill_between twice for alternating gray tones
            for _ in range(2 if i % 2 == 0 else 1):
                ax.fill_between(x=[self.output_grid[i - 1], self.output_grid[i]],
                                y1=ymin, y2=ymax, alpha=.1, color='gray')

        ax.hlines([.25, .5, .75, 1], xmin=0, xmax=max(self.output_grid),
                  color='black', linewidth=.5, linestyle=(0, (5, 10)))

        # plots should be on right edge of each bucket
        next_step = survival_curves.columns[-1] + \
            np.diff([survival_curves.columns[-2],
                     survival_curves.columns[-1]])[0]
        timepoints_shifted = [*survival_curves.columns, next_step]

        # plot individuals
        for i in range(survival_curves.shape[0]):
            curr_color = next(ax._get_lines.prop_cycler)['color']

            ax.plot(timepoints_shifted,
                    [1, *survival_curves.iloc[i, :]],
                    drawstyle="steps-pre",
                    alpha=.8,
                    marker='o',
                    markersize=2,
                    color=curr_color,
                    label=survival_curves.index[i]
                    )

            # plot hazards
            if include_hazards:

                ax2.bar(
                    x=hazards.columns + 10 * (i + 1),
                    height=hazards.iloc[i, :],
                    width=5,
                    align='edge',
                    alpha=.4,
                )

            # plot event time
            if Y is not None:
                durations = Y[defaults.DURATION_COL]
                events = Y[defaults.EVENT_COL]
                ax.annotate(text="",
                            xy=(durations[i], 1.1),
                            xytext=(durations[i], 1.2),
                            arrowprops=dict(
                                facecolor=curr_color,
                                alpha=1 if events[i] else .2,
                                linestyle='solid' if events[i] else '--',
                                shrink=0.1,
                            ),
                            )

        ax.legend()
        return ax

    @staticmethod
    def _predict_survival_curves(event_rates):
        """
        Since the prediction yields event rates, they can be transformed to a
        survival function

        Returns
        ---
        df of survival curve

        Notes
        -----
        Survival function is implemented as:

        .. math:: S(t|x) = \\prod_{l:l\\leq l_i} (1 - h_l^i)
        """
        if isinstance(event_rates, np.ndarray):
            event_rates = pd.DataFrame(event_rates)

        surv_curve = 1 - event_rates
        for i in range(1, event_rates.shape[1]):  # i=0 remains unchanged!
            surv_curve.iloc[:, i] = surv_curve.iloc[:, i - 1] * \
                surv_curve.iloc[:, i]

        return surv_curve

    # Validation
    @ staticmethod
    def _validate_list(obj):
        if isinstance(obj, (int)):
            return [obj]
        elif isinstance(obj, (list, pd.Series, np.ndarray)):
            return obj
        elif "Hashable" in obj.__class__.__name__:
            return np.array(list(obj.values()))
        else:
            return obj

    def _validate_X(self, X):
        if not isinstance(X, (np.ndarray, tf.Tensor)):
            try:
                X = np.array(X)
            except:
                raise TypeError("X should be a numpy array or tensor.")
        if np.any(np.isnan(X)):
            raise ValueError("X must not contain NaN.")
        if X.ndim != 2:
            raise ValueError(
                f"X should be of shape (n_samples, n_features), not {X.shape}")
        if not isinstance(X, tf.Tensor):
            try:
                X = tf.convert_to_tensor(X)
            except (TypeError, RuntimeError, ValueError):
                raise ValueError(
                    f"X could not be converted to tf.tensor: {type(X)}.")

        return X

    def _validate_y(self, y):
        if not isinstance(y, np.ndarray):
            try:
                y = np.array(y)
            except:
                raise TypeError("y should be a numpy array.")
        if np.any(np.isnan(y)):
            raise ValueError("y must not contain NaN.")
        if (y.ndim != 2):
            raise ValueError("y should have 2 dimensions, not", y.ndim)
        if (y.shape[1] != 2):
            raise ValueError(
                "y should have two columns in axis 1, not", y.shape[1])

        max_t = max(self.output_grid)
        if any(y[:, 0] > max_t):
            if self.verbose:
                warnings.warn(
                    f"labels contain {(y[:, 0] > max_t).sum()} events beyond output grid."
                    "They will be clipped and censored.")
            y[y[:, 0] > max_t] = [max_t + 1, 0]

        return y

    def _validate_fit_args(self):
        # learning
        if not (0 <= self.validation_size < 1):
            raise ValueError(
                "validation_size should be between 0 and 1 (exclusive)")

        elif (self.validation_size == 0) and self.use_early_stopping:
            raise ValueError("Early stopping requires validation_size > 0")

    def _create_loss_fn(self):
        return DrsaLoss(self.alpha,
                        self.beta,
                        self.output_grid,
                        self.loss_normalize,
                        self.loss_normalize_buckets).calc

    def _set_output_grid(self, y):

        if isinstance(y, pd.DataFrame):
            durations = y[defaults.DURATION_COL]
        else:
            durations = pd.Series(y[:, 0], name=defaults.DURATION_COL)

        max_duration = durations.quantile(self.output_grid_max_quantile)
        if self.output_grid_type == 'linear':
            self.output_grid = np.linspace(
                0,
                max_duration,
                self.output_grid_num_nodes + 1)[1:]

        elif self.output_grid_type == 'log':
            self.output_grid = np.logspace(
                np.log10(10),
                np.log10(max_duration),
                self.output_grid_num_nodes + 1)[1:]

        elif self.output_grid_type == 'quantile':
            self.output_grid = durations.quantile(
                np.linspace(0, 1, self.output_grid_num_nodes + 1)).values[1:]
        else:
            raise NotImplementedError(f"'{self.output_grid_type} not implemented!'")

        self.output_grid = np.round(self.output_grid, 0)

    def fit(self, y):

        self._set_output_grid(y)

        self._validate_fit_args()

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipvalue=self.adam_clipvalue
        )

        self.loss_fn = self._create_loss_fn()

        self.model_ = self._create_model()

        self.model_.compile(self.optimizer, loss=self.loss_fn)

        if self._develop_mode:
            self.model_.run_eagerly = True

        # Callbacks
        self._callbacks = [TerminateOnNaN()]

        if self.verbose:
            self._callbacks.append(TqdmCallback(verbose=1))

        if self.use_early_stopping:
            self._callbacks.append(
                EarlyStopping(monitor='val_loss',
                              mode='min',
                              verbose=self.verbose,
                              min_delta=self.early_stopping_min_delta,
                              patience=self.early_stopping_patience,
                              restore_best_weights=True
                              ))

    def get_initial_bias(self, y):
        print("setting initial bias ")
        if self.beta != 0:
            if self.verbose:
                warnings.warn(
                    "initial bias estimation not implemented for beta != 0. ignoring beta")
        if self.loss_normalize_buckets:
            if self.verbose:
                warnings.warn("initial bias not implemented for loss normalized by buckets.")

        if type(y) == pd.DataFrame:
            y = y.to_numpy()
        grid = np.append(self.output_grid, max(self.output_grid.max() + 1, y[:, 0].max() + 1))
        epsilon = np.finfo(np.float32).eps
        y_unc = y[:, 1] == 1
        y_unc = y[y_unc]
        y_cens = y[:, 1] == 0
        y_cens = y[y_cens]

        events_per_bucket, _ = np.histogram(y_unc[:, 0], bins=grid)

        # substract events in bucket since the receive a different loss
        events_after_bucket = np.cumsum(events_per_bucket[::-1])[
            ::-1] - events_per_bucket

        censored_per_bucket, _ = np.histogram(y_cens[:, 0], bins=grid)
        censored_in_or_after_bucket = np.cumsum(censored_per_bucket[::-1])[::-1]
        # what to do about patients in the last bucket?? maybe change this TODO

        def logit(prob):
            return -np.log(((1 - prob) / (prob + epsilon)) + epsilon)

        bias_probs = self.alpha * events_per_bucket / \
            (self.alpha * events_per_bucket + self.alpha * events_after_bucket
             + (1 - self.alpha) * censored_in_or_after_bucket + epsilon)
        bias = logit(bias_probs)
        expected_loss = np.sum(
            - self.alpha * events_per_bucket * np.log(bias_probs + epsilon)
            - self.alpha * events_after_bucket * np.log(1 - bias_probs + epsilon)
            - (1 - self.alpha) * censored_in_or_after_bucket * np.log(1 - bias_probs + epsilon)
        )

        if self.loss_normalize:
            expected_loss = expected_loss / len(y)

        print("initial bias:")
        print(bias)
        print("excpected loss for good bias initialization")
        print(expected_loss)
        return bias


class DrsaLoss():
    """
    Loss from Ren et al 2018: Deep Recurrent Survival Analysis
    """

    def __init__(self,
                 alpha,
                 beta,
                 output_grid,
                 normalize=False,
                 normalize_buckets=False,
                 use_ipcw=False,
                 ipcw_max_factor=10
                 ):
        """
        Args:
            alpha (float): Specify relative weight between L_z and L_c
            beta (float): Specify relative weight between L_c (uncensored)
                and L_c (censored)
            output_grid (enumerable): output timepoints of prediction times
            normalize (bool): Should loss be normalized by n?
            normalize_buckets (bool): Should loss be normalized by #buckets
            use_ipcw (bool): Use Inverse Probability of censoring weighting
        """
        # validate and set internal variables
        if not (0 <= alpha <= 1):
            raise ValueError("alpha should be between 0 and 1")
        self.alpha = alpha

        if not beta >= 0:
            raise ValueError("beta should be greater or equal to 0")
        self.beta = beta

        self.output_grid = np.array(output_grid)
        self.normalize = normalize
        self.normalize_buckets = normalize_buckets
        self.use_ipcw = use_ipcw
        self.ipcw_max_factor = ipcw_max_factor

    _epsilon = np.finfo(float).eps

    def _validate_input(self, y):
        if isinstance(y, tf.Tensor):
            y = tf.cast(y, dtype='float32')
        else:
            y = tf.convert_to_tensor(y, dtype='float32')
        return y

    def calc(self, y_act, y_pred):
        """calculate drsa loss

        Args:
            y_act (tensor): Target with columns (event_time, has_event)
            y_pred (tensor): Predicted hazards (n_samples, len(output_grid))
        """

        y_act = self._validate_input(y_act)
        y_pred = self._validate_input(y_pred)

        loss_z = 0
        loss_c = 0

        if self.alpha > 0:
            loss_z += self._calc_loss_z(y_act, y_pred)

        if self.alpha < 1:
            loss_c += self._calc_loss_c(y_act, y_pred)

        loss = self.alpha * loss_z + (1 - self.alpha) * loss_c

        if self.normalize_buckets:
            output_weights = tf.reduce_sum(
                tf.cast(self._create_pred_mask_upto_li(y_act, include_li=True),
                        'float32'),
                axis=1)
            loss = tf.divide(loss, output_weights)

        if self.use_ipcw:
            # 1. calc ipcw on population of censored individuals
            y_act_censored = y_act[y_act[:, 1] == 0]

            if len(y_act_censored > 0):

                censoring_km = (
                    KaplanMeierFitter()
                    .fit(y_act_censored[:, 0])
                )

                # take inverse of all event times and scale up loss
                ipcw = 1 / censoring_km.survival_function_at_times(y_act[:, 0])
                ipcw = np.clip(ipcw, None, self.ipcw_max_factor)
                loss = loss * ipcw

        if self.normalize:
            if self.use_ipcw:
                loss = loss / ipcw.sum()
            else:
                loss = loss / len(y_act)

        return tf.reduce_sum(loss)

    def _calc_loss_c(self, y_act, y_pred):
        """calculate categorical loss between y_pred and y_act

        Args:
            y_act (tensor): Target with columns (event_time, has_event)
            y_pred (tensor): Predicted outcome (n_samples, len(output_grid))

        Remark:
            Can handle censored and uncensored y_act

        """

        # create prediction for actual bucket
        is_censored = y_act[:, 1] == 0

        mask = self._create_pred_mask_upto_li(y_act, include_li=True)

        surv = tf.reduce_prod(1 - y_pred * mask, axis=1)

        # depending on censoring, calculate loss
        loss_cen = tf.math.log(surv + self._epsilon)
        loss_unc = tf.math.log(1 - surv + self._epsilon)

        loss_c = tf.where(is_censored, loss_cen, self.beta * loss_unc)

        return -loss_c

    def _calc_loss_z(self, y_act, y_pred):
        """calculate z loss between y_pred and y_act

        Args:
            y_act (tensor): Target with columns (event_time, has_event)
            y_pred (tensor): Predicted outcome (n_samples, len(output_grid))
        """

        is_censored = y_act[:, 1] == 0

        # difference in correct bucket
        loss1 = tf.math.log(self._epsilon + self._get_h_lis(y_act, y_pred))

        # difference to zero in preceeding buckets
        mask = self._create_pred_mask_upto_li(y_act, include_li=False)

        loss2 = tf.reduce_sum(
            tf.math.log(1 - (y_pred * mask) + self._epsilon), axis=1)

        loss_z = -(loss1 + loss2)

        loss_z = tf.where(is_censored, 0, loss_z)

        return loss_z

    def _create_pred_mask_upto_li(self, y_act, include_li):
        """creates mask for prediction to mask in only up to l_i

        Args:
            y_act (tensor): Target with columns (event_time, has_event)
            include_li (bool): Should the (l_i)th element be included in the mask?
        """
        # get l_i-th prediction per row from y_act
        lis = self._get_lis(y_act)

        mask = np.zeros(
            shape=(
                y_act.shape[0],
                len(self.output_grid)),
            dtype=np.bool)

        for i, l_i in enumerate(lis):
            idx = l_i + 1 if include_li else l_i
            mask[i, :idx] = 1

        result = mask.reshape(len(y_act), len(self.output_grid))

        return result

    def _get_h_lis(self, y_act, y_pred):
        """returns the l_is (prediction in actual bucket) for each individual

        Args:
            y_act (tensor): Target with columns (event_time, has_event)
            y_pred (tensor): Predicted outcome (n_samples, len(output_grid))
        """

        return tf.reduce_sum(
            tf.one_hot(
                self._get_lis(y_act),
                depth=len(self.output_grid)
            ) * y_pred,
            axis=1
        )

    def _get_lis(self, y_act):
        """
        Returns l_i bucket index for each individual

        Example
        ---
        output_grid of [0, 100, 200]:

        _get_lis([50, 100, 150, 200, 250]) should return [0, 0, 1, 1, 2]
        """
        return tf.raw_ops.Bucketize(
            input=y_act[:, 0],
            boundaries=list(self.output_grid + 0.001)) - 1


class NBatchLogger(Callback):
    def __init__(self, display=100):
        '''
        display: Number of batches to wait before outputting loss
        '''
        self.seen = 0
        self.display = display

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            print('\n{0}/ - Batch Loss: {1}'.format(self.seen, logs.get('loss')))
