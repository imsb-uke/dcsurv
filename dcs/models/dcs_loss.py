import tensorflow as tf
from dcs.util import hazard_to_survival


class DcsLoss:
    """Calculate loss following Kamran et al. 2021 that combines
       L_RPS and L_Kernel
    """

    def __init__(self, output_grid, lambda_=1, sigma=1,
                 normalize=True, kernel_include_censored=True,
                 y_pred_is_hazard=False):
        """Initializes class that provides loss_fn

        Args:
            output_grid (list like): output node timepoints
            lambda (float): weighting factor between L_RPS and L_kernel
            sigma (float): spreading factor for L_kernel
            normalize (bool): should loss be normalized by n_individuals?
            y_pred_is_hazard (bool): interpret y_pred as hazard or direct survival estimate?
        """

        if output_grid[0] != 0:
            raise ValueError("output is not 0 at first node!")

        # self.output_grid = tf.convert_to_tensor(output_grid)
        self.output_grid = output_grid

        if sigma < 0:
            raise ValueError(f"sigma should be a positive real number, not {sigma}")
        self.sigma = sigma

        if lambda_ < 0:
            raise ValueError(f"lambda_ should be a positive real number, not {lambda_}")
        self.lambda_ = lambda_

        self.normalize = normalize

        self.y_pred_is_hazard = y_pred_is_hazard

        self.kernel_include_censored = kernel_include_censored

    def _loss_rps_unc(self, y_true, y_pred):

        # remove first output grid node == 0
        t_le_zi = tf.cast(
            self.output_grid_repeated[:, 1:] < self.zi_repeated,
            dtype=tf.float32)

        return tf.square(y_pred - t_le_zi)

    def _loss_rps_cen(self, y_true, y_pred):
        """Calculates RPS censored loss part

        Remark:
            Decision to exclude censoring bucket from loss, only count those where the patient
            survived the whole bucket
        """

        # for censoring, shift grid one to the right. so remove last node
        survived_buckets_mask = tf.cast(
            self.output_grid_repeated[:, 1:] < self.zi_repeated,
            tf.float32)

        return tf.square(y_pred - 1) * survived_buckets_mask

    def _loss_kernel(self, y_true, y_pred):

        grid_no_zero = self.output_grid[1:]

        # calc uncensored pairs
        is_unc = tf.expand_dims(y_true[:, 1], axis=1)

        if self.kernel_include_censored:
            censoring_mask = tf.repeat(tf.cast(is_unc, tf.bool), tf.shape(is_unc)[0], axis=1)
        else:
            censoring_mask = tf.cast(is_unc * tf.transpose(is_unc), tf.bool)

        # calc event time bucket indices
        zi_idxs_to_extract = tf.searchsorted(
            tf.cast(grid_no_zero, tf.float32),
            self.zi,
            side='left')
        # remember zis that are beyond output grid
        zi_bigger_max_mask = (zi_idxs_to_extract == self.n_outputs)
        zi_idxs_to_extract = tf.clip_by_value(
            zi_idxs_to_extract, clip_value_min=0, clip_value_max=self.n_outputs - 1)

        # project zi to grid to avoid comparing inside same bucket
        zi_projected = tf.cast(
            tf.expand_dims(
                tf.gather(self.output_grid, zi_idxs_to_extract + 1),
                axis=1), tf.float32)

        # replace those that were beyond output grid with max value
        zi_projected = tf.transpose(
            tf.where(
                zi_bigger_max_mask,
                tf.float32.max,
                tf.transpose(zi_projected)
            )
        )

        zi_repeated = tf.repeat(zi_projected, tf.shape(zi_projected)[0], axis=1)
        zj_repeated = tf.transpose(zi_repeated)
        zi_le_zj = zi_repeated < zj_repeated
        Bij = tf.logical_and(censoring_mask, zi_le_zj)

        # calc S_zi_xi
        mask_zi_xi = tf.one_hot(zi_idxs_to_extract, self.n_outputs)
        S_zi_xi = tf.reduce_sum(y_pred * mask_zi_xi, axis=1)
        S_zi_xi_repeated = tf.repeat(
            tf.expand_dims(S_zi_xi, axis=1),
            repeats=tf.shape(self.zi)[0], axis=1)

        # calc S_zi_xj
        S_zi_xj_repeated = tf.transpose(
            tf.gather(y_pred, zi_idxs_to_extract, axis=1)
        )

        # calc loss
        result = tf.exp(
            -(S_zi_xj_repeated - S_zi_xi_repeated) / self.sigma
        )

        return tf.where(Bij, result, 0)

    def loss_fn(self, y_true, y_pred):
        """Actual loss function with initially defined parameters.

        Args:
            y_true (n x 2 tensor): Actual survival information
            y_pred (n x n_outputs tensor): Predicted survival information for n_outputs time stamps

        Returns:
            1x1 tensor: calculated loss
        """

        if self.y_pred_is_hazard:
            y_pred = hazard_to_survival(y_pred)

        n_individuals = tf.shape(y_true)[0]

        # -1 to exclude first node with t_0 = 0
        self.n_outputs = tf.shape(self.output_grid)[0] - 1

        self.output_grid_repeated = tf.cast(
            tf.repeat(
                tf.expand_dims(self.output_grid, axis=0),
                n_individuals, axis=0),
            tf.float32)

        self.is_censored_mask = tf.repeat(
            tf.expand_dims(y_true[:, 1] == 0, axis=1),
            self.n_outputs, axis=1
        )

        self.zi = tf.cast(y_true[:, 0], tf.float32)

        self.zi_repeated = tf.repeat(
            tf.expand_dims(self.zi, axis=1),
            self.n_outputs, axis=1)

        # L_RPS
        loss_rps_unc = self._loss_rps_unc(y_true, y_pred)
        loss_rps_cen = self._loss_rps_cen(y_true, y_pred)
        loss_rps = tf.where(self.is_censored_mask,
                            loss_rps_cen,
                            loss_rps_unc)

        # L_kernel
        if self.lambda_ > 0:
            loss_kernel = self._loss_kernel(y_true, y_pred)
        else:
            loss_kernel = tf.zeros_like(loss_rps)

        # aggregate losses on axis1 to get one value per individual
        if self.normalize:
            # normalize loss rps by number of output steps and number of individuals
            loss_rps_agg = tf.reduce_mean(loss_rps, axis=[0, 1])

            # normalize kernel by number of non zero entries
            n_kernel_comparisons = tf.math.count_nonzero(loss_kernel, dtype=tf.float32)

            loss_kernel_agg = tf.reduce_sum(loss_kernel, axis=[0, 1]) / \
                tf.math.maximum(tf.constant(1, tf.float32), n_kernel_comparisons)
        else:
            loss_rps_agg = tf.reduce_sum(loss_rps, axis=[0, 1])
            loss_kernel_agg = tf.reduce_sum(loss_kernel, axis=[0, 1])

        # aggregate all individuals
        loss = tf.reduce_sum(
            loss_rps_agg + self.lambda_ * loss_kernel_agg
        )

        return loss
