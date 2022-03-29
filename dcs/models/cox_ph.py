from sklearn.base import BaseEstimator
from lifelines.fitters import coxph_fitter
import numpy as np
import pandas as pd


class CoxPH(BaseEstimator):

    def __init__(self,
                 penalizer=0.0001,
                 l1_ratio=0,
                 step_size=0.5,
                 initial_point=None,
                 robust=False,
                 verbose=False):
        """
        Args:
            penalizer (float): Assign penalty to size of beta vector
            l1_ratio (float): Weight between l2 and l1 in penalizer
            step_size (float): step initial step size of CoxPH fitting algorithm
            initial_point (float): Point from where to start CoxPH fitting algorithm
            verbose (bool): print additional information
        Returns:
            lifelines' CoxPH model
        """
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.step_size = step_size
        self.model = coxph_fitter.CoxPHFitter(penalizer=self.penalizer, l1_ratio=l1_ratio)
        self.verbose = verbose
        self.initial_point = initial_point
        self.robust = robust

    def prepare_input(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns

        if np.isnan(X).any().any():
            raise ValueError('X cannot contain NaN')

        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        if (y is not None) and (np.isnan(y).any()):
            raise ValueError('y cannot contain NaN')

        return X, y

    def fit(self, X, y, **fit_kwargs):

        X, y = self.prepare_input(X, y)

        df = pd.DataFrame(X.copy())
        df[['duration', 'events']] = y

        self.model.fit(df,
                       duration_col='duration',
                       event_col='events',
                       show_progress=self.verbose,
                       step_size=self.step_size,
                       initial_point=self.initial_point,
                       robust=self.robust,
                       **fit_kwargs
                       )

        if self.verbose:
            self.model.print_summary()

        return self

    def predict(self, X):
        X, y = self.prepare_input(X)
        return self.model.predict_survival_function(X).T
