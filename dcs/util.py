
import pandas as pd
import numpy as np


def interpolate_prediction(df,
                           target_grid,
                           interpolate_method='pad',
                           interpolate_order=3):
    """
    Interpolates df assuming columns are days

    interpolate_method can be 'pad', 'linear', 'quadratic', 'cubic', spline'
    """
    def interpolate(df: pd.DataFrame,
                    step_size=1,
                    method='linear'):
        """Interpolates pandas df to time steps of one on index

        Args
        ----
        df (df): dataframe to interpolate in rows, cols are observations
        method (string): {'linear', 'ffill'}
        """
        if step_size is None:
            return df

        return (df
                .reindex(range(0, max(df.index.astype('int64')) + 1, step_size))
                .interpolate(method=method)
                )

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

    if 0 not in final_grid:
        final_grid = [0, *final_grid]

    # readd original points
    final_grid = sorted(list(set(final_grid).union(set(df.columns))))

    return df_interp.loc[:, final_grid]


def hazard_to_survival(haz):

    def shift_ax1(tensor, places=1, fill_value=1, shift_left=True):

        import tensorflow as tf

        if places == 0:
            return tensor
        if places >= tensor.get_shape()[1]:
            return tf.ones_like(tensor)

        if len(tensor.shape) != 2:
            raise ValueError("tensor should have 2 dims")

        append_tensor = (tf.ones_like(tensor) * fill_value)[:, :places]

        if shift_left:
            to_concat = (tensor[:, places:], append_tensor)
        else:
            to_concat = (append_tensor, tensor[:, :-places])

        return tf.concat(to_concat, axis=1)

    tmp_factor = 1 - haz
    result = tmp_factor
    factors_shifted = shift_ax1(
        tmp_factor,
        fill_value=1,
        shift_left=False,
        places=1)

    n_iterations = haz.shape[1] - 1
    for i in range(n_iterations):
        result = result * factors_shifted
        factors_shifted = shift_ax1(
            factors_shifted,
            fill_value=1,
            shift_left=False,
            places=1)

    return result
