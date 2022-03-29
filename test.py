
import sys
sys.path.append('dcs')
import dcs

import pandas as pd

# %%
config = dcs.util.load_yaml('config.yaml')

# %% [markdown]
# # Load Dataset

# %%
pipeline = dcs.pipelines.get_pipeline(config['dataset'])

# %%
dataset = dcs.datasets.get_dataset(config['dataset'])
# display(dataset.head())

train_X, train_y, test_X, test_y = dcs.preprocessing.train_test_split_X_y(
    dataset,
    random_state=config['random_seed'],
    test_size=config['test_size'])

train_X_t = pipeline.fit_transform(train_X)
test_X_t = pipeline.transform(test_X)

# %% [markdown]
# # Model

# %%
model = dcs.models.CoxPH(verbose=True)
model.fit(train_X_t, train_y)

train_pred = model.predict(train_X_t)

train_pred.sample(10).T.plot()

# %%
test_pred = model.predict(test_X_t)

results = pd.Series({
    "c-index-td": dcs.evaluation.concordance_index_td(test_y, test_pred),
    "ddc": dcs.evaluation.ddc(test_y, test_pred),
}, name='CoxPH')

# display(results.to_frame())

# %%

from dcs import defaults
import numpy as np
from sksurv.metrics import cumulative_dynamic_auc


target = test_y
prediction = test_pred

tau_1 = None
tau_2 = None

n_timesteps = 50

decimals = 3


# %%

def __get_predictions_at(predictions, t):
    bucket_of_t = np.digitize(t, predictions.columns[:-1])
    return predictions.iloc[:, bucket_of_t]


def reshape_df_records(df):
    return (
        df
        .assign(
            has_event=lambda df: df[defaults.EVENT_COL] == 1)
        .iloc[:, [1, 0]]
        .to_records(index=False)
    )


# if train_y is not provided, use target to estimate KM distribution
train_y_records = reshape_df_records(target if train_y is None else train_y)
test_records = reshape_df_records(target)

# make sure predictions are only inside target
timesteps = np.linspace(
    tau_1 or target.loc[lambda df: df[defaults.EVENT_COL] == 1, defaults.DURATION_COL].min(),
    tau_2 or target.loc[lambda df: df[defaults.EVENT_COL] == 1, defaults.DURATION_COL].max(),
    num=n_timesteps + 1)[:-1]

prediction_at = __get_predictions_at(prediction, timesteps)

prediction_at.columns = timesteps

auc_t, cdauc = cumulative_dynamic_auc(
    survival_train=train_y_records,
    survival_test=test_records,
    estimate=(1 - prediction_at).to_numpy(),
    times=timesteps,
    tied_tol=10**(-decimals),
)


# %% [markdown]
# # Load Models
