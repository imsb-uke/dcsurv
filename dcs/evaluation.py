from pycox.evaluation import EvalSurv
import numpy as np
import pandas as pd
from dcs import defaults
from sksurv.metrics import cumulative_dynamic_auc


def concordance_index_td(target, predictions):
    """
    Time dependent concordance index
    """
    _check_is_comparable(target, predictions)

    if isinstance(target, pd.DataFrame):
        durations = target.iloc[:, 0].to_numpy()
        events = target.iloc[:, 1].to_numpy()
    elif isinstance(target, np.ndarray):
        durations = target[:, 0]
        events = target[:, 1]

    ev = EvalSurv(predictions.T, durations, events, 'km')
    return ev.concordance_td()


def d_calibration(target, prediction, n_buckets=10, return_b=False,
                  return_b_event_censored=False,
                  chi2_alpha=.05):
    """
    d-calibration score
    """
    # for each patient, get survival curve at time of event
    bucket_survival_limits = prediction.columns   # e.g. 0,10,20,30,40

    # epsilon to avoid div by 0
    __epsilon = np.finfo(float).eps

    # leave out last bucket, so patients living longer than that still
    # get assigned to last bucket
    bucket_each_patient = np.digitize(target[defaults.DURATION_COL], bucket_survival_limits[:-1])
    surv_probability_each_patient = np.zeros(len(prediction))

    for idx, b in enumerate(bucket_each_patient):
        surv_probability_each_patient[idx] = prediction.iloc[idx, b]

    buckets = np.round(np.linspace(0, 1, n_buckets + 1), 3)
    buckets[-1] = 1.1
    b_event = np.zeros(n_buckets)
    b_censored = np.zeros(n_buckets)
    for k in range(len(buckets) - 1):   # 0 0.25 0.5 0.75 1
        p_k = buckets[k]
        p_k1 = buckets[k + 1]
        patients_in_bucket = ((surv_probability_each_patient >= p_k)
                              & (surv_probability_each_patient < p_k1))

        eq1 = np.sum(patients_in_bucket & (target[defaults.EVENT_COL] == 1))

        is_censored = (target[defaults.EVENT_COL] == 0).to_numpy()
        censored_patients_in_bucket = (patients_in_bucket * is_censored)

        eq2 = (
            (surv_probability_each_patient - p_k)
            / np.clip(surv_probability_each_patient, __epsilon, None)
        )[censored_patients_in_bucket == 1].sum()

        eq3 = ((p_k1 - p_k) / np.clip(surv_probability_each_patient, __epsilon, None))[
            (surv_probability_each_patient >= p_k1) & (is_censored == 1)].sum()

        b_event[k] = eq1
        b_censored[k] = np.nansum((eq2, eq3))

    b = b_event + b_censored
    b_event_normalized = b_event / len(prediction)
    b_censored_normalized = b_censored / len(prediction)
    b_normalized = b / len(prediction)

    # error prediction
    expected_value = 1 / n_buckets
    mse = np.sum((expected_value - b_normalized)**2)

    if return_b_event_censored:
        return b_event_normalized, b_censored_normalized
    elif return_b:
        return mse, b, b_normalized
    else:
        return mse


def ddc(target, prediction, **d_calibration_kwargs):
    """
    Sources
    ---
    Kamran et al. 2021
    https://towardsdatascience.com/kl-divergence-python-example-b87069e4b810
    """

    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    d_calibration_kwargs.pop('return_b', None)

    d_calibration_buckets = d_calibration(
        target, prediction, **d_calibration_kwargs, return_b=True)[2]

    n = len(d_calibration_buckets)

    return kl_divergence(d_calibration_buckets, np.ones(n) / n)


def cdauc(target, prediction,
          train_y=None,
          tau_1=None,
          tau_2=None,
          n_timesteps=50,
          decimals=3):

    def reshape_df_records(df):
        return (
            df
            .assign(
                **{defaults.EVENT_COL: lambda df: df[defaults.EVENT_COL] == 1})
            .iloc[:, [1, 0]]
            .to_records(index=False)
        )

    # if train_y is not provided, use target to estimate KM distribution
    train_y_records = reshape_df_records(target if train_y is None else target)
    target_records = reshape_df_records(target)

    # make sure predictions are only inside target
    timesteps = np.linspace(
        tau_1 or target.loc[lambda df: df[defaults.EVENT_COL] == 1, defaults.DURATION_COL].min(),
        tau_2 or target.loc[lambda df: df[defaults.EVENT_COL] == 1, defaults.DURATION_COL].max(),
        num=n_timesteps + 1)[:-1]

    prediction_at = __get_predictions_at(prediction, timesteps)

    prediction_at.columns = timesteps

    auc_t, cdauc = cumulative_dynamic_auc(
        survival_train=train_y_records,
        survival_test=target_records,
        estimate=(1 - prediction_at).to_numpy(),
        times=prediction_at.columns,
        tied_tol=10**(-decimals),
    )

    return cdauc


def __get_predictions_at(predictions, t):
    bucket_of_t = np.digitize(t, predictions.columns[:-1])
    return predictions.iloc[:, bucket_of_t]


def _check_is_comparable(target, predictions):
    """Does the prediction match the target?
    """

    if isinstance(target, pd.DataFrame) and isinstance(predictions, pd.DataFrame):
        if not predictions.sort_index().index.equals(target.sort_index().index):
            raise ValueError("target and predictions should have same index!")
    else:
        # on numpy arrays, at least the length should be the same
        if target.shape[0] != predictions.shape[0]:
            raise ValueError("target and predictions should have the same length!")
