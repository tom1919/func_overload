# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 00:53:01 2022

Class for optimizing portfolio weights

@author: tommy
"""

#%% libs

import re
import logging
import cvxpy as cp
import numpy as np
import pandas as pd
from .utils import todate
from abc import ABC, abstractmethod
from pydantic import BaseModel, PrivateAttr

#%% opt 

class Opt(BaseModel):
    '''
    Optimize portfolio weights as desribed by the attributes.
    
    Attributes
    ----------
    rtn : pd.DataFrame
        yhat column is the forecasted ranking of returns for each ticker.
    cov : pd.DataFrame
        covariance matrix for the tickers.
    cur_wts : pd.DataFrame
        current weights used to constrain turnover.
    benchmark : str, optional
        the benchmark ticker. The default is 'SPY'.
    opt_kind : str, optional
        if 'max_alpha' then objective is to maximize the weighted return rankings
        with risk constraint <= the benchmark risk. if min_risk then the 
        objective is to minimize risk with weighted sum of return rankings 
        (exposure) >= benchmark ranking. The default is 'max_alpha'.
    bm_min : float, optional
        constraint for minimum weight of the benchmark. The default is .8.
    max_wt : float, optional
        constraint for max wt of the other tickers. The default is .05.
    max_cash_wt : float, optional
        constraint for max wt for cash. The default is .07.
    min_nonzero_wt : float, optional
        constraint for the min weight if it is non zero. This prevents the 
        optimizer from returning small non-investable solution wts The 
        default is .01.
    solver : str, optional
        the solver that CVXPY uses. The default is 'XPRESS'.
    verbose : bool, optional
        print optimization info. The default is True.
    lg : logging.logger
        logger used for warnings
    '''
    
    rtn: pd.core.frame.DataFrame 
    cov: pd.core.frame.DataFrame 
    cur_wts: pd.core.frame.DataFrame = None
    benchmark = 'SPY' 
    opt_kind = 'MaxAlpha'
    bm_min = .8 
    max_wt = .05  
    max_cash_wt = .07
    min_nonzero_wt = .01 
    solver = 'XPRESS' 
    verbose = False
    lg: logging.Logger = None
    
    _w: cp.expressions.variable.Variable = PrivateAttr()
    _b: cp.expressions.variable.Variable = PrivateAttr()
    _alpha: cp.atoms.affine.binary_operators.MulExpression = PrivateAttr()
    _risk: cp.affine.binary_operators.MulExpression = PrivateAttr()
    _alpha_limit: float = PrivateAttr()
    _risk_limit: float = PrivateAttr()
    _constraints: list = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def solve(self):
        '''
        Solves the optimization problem

        Returns
        -------
        opt : pd.DataFrame
            the rtn DF with a column for solution weights added.
        sol_meta : dict
            the exposure, risk of the opt wts and limits for exposure/risk used.
        '''
        self._add_logger()
        self.rtn, self.cov = self._conform_opt_inputs()
        self._base_variables()
        self._base_constraints()
        subclass = self._get_subclass('opt_kind')
        prob, self = subclass._define_prob(self)
        prob.solve(verbose = self.verbose, solver = self.solver)
 
    
    def _add_logger(self):
        if self.lg is None:
            from thelogger import lg
            self.lg = lg

    def _conform_opt_inputs(self):
        '''
        Retruns versions of the rtn and cov DFs that have the same columns in 
        the same order
        '''
        rtn2, cov2, lg = self.rtn.copy(), self.cov.copy(), self.lg
        
        dt = todate(rtn2.date.unique()[0])
        if rtn2.isna().any().any():
            rtn2 = rtn2.dropna()
            dropped_rtn = set(self.rtn.ticker) - set(rtn2.ticker)
            lg.w(f'tickers from {dt:%Y-%m-%d} rtn dropped bc there was NAs: '\
                 f'{dropped_rtn}')
        if cov2.isna().all().any():
            non_na = cov2.columns[~cov2.isna().all()]
            cov2 = cov2.loc[non_na, non_na]
            dropped_cov = set(self.cov.columns) - set(cov2.columns)
            lg.w(f'tickers from {dt:%Y-%m-%d} cov dropped bc there was NAs: '\
                 f'{dropped_cov}')
        
        if set(rtn2.ticker) != set(cov2.columns):
            uni_diff = set(rtn2.ticker) ^ set(cov2.columns)
            lg.debug(f"rtn and cov {dt:%Y-%m-%d} universe doesn't match: "\
                     f"{uni_diff}")
            
        uni = list(set(rtn2.ticker).intersection(set(cov2.columns)))
        uni.sort()    
        rtn2 = rtn2.loc[rtn2.ticker.isin(uni)]
        rtn2 = rtn2.sort_values('ticker')
        cov2 = cov2.loc[uni, uni]
        
        return rtn2, cov2 

    def _base_variables(self):
        n  = len(self.rtn)
        self._w, self._b = cp.Variable(n), cp.Variable(n, boolean=True) 
        mu = self.rtn.yhat.to_numpy().reshape([n, 1])
        self._alpha = mu.T @ self._w
        sigma = self.cov.to_numpy()
        self._risk = cp.quad_form(self._w, sigma)

    def _base_constraints(self):
        '''
        Constraints that are commom to the different kinds of optimizations
    
        Notes
        -----
        b : cvxpy.Variable
            binary variable used to constrain non-zero weights to be greater 
            than a specified amount. cvxpy doesn't support semi-continuous vars, 
            and this approach for defining the constraint makes this a MIQP
        '''
        
        w = self._w
        b = self._b
        r = self.rtn
        n  = len(r)
        
        min_wts = np.where(r.ticker == self.benchmark, self.bm_min, 0)
        max_wts = np.where(r.ticker == self.benchmark, 1, self.max_wt)
        max_wts = np.where(r.ticker == 'cash', self.max_cash_wt, max_wts)
        
        c = [
            cp.sum(w) == 1,
            w >= min_wts,
            w <= max_wts,
            w >= cp.multiply(b, np.ones(n) * self.min_nonzero_wt),
            w <= cp.multiply(b, np.ones(n))
            ]
        
        self._constraints = c

    def _get_subclass(self, str_keyword):
        
        subclasses = _Interface.__subclasses__()
        pattern = "<class '__.+__._(.+)'>"
        str_subclasses = list(map(lambda x: re.search(pattern, str(x)).group(1), 
                                  subclasses))
        
        kind = getattr(self, str_keyword)
        try:
            str_subclasses = list(str_subclasses)
            i = str_subclasses.index(kind)
        except ValueError:
            err_msg = f"'valid args for {str_keyword} are: {str_subclasses}"
            raise NotImplementedError(err_msg)
            
        subclass = subclasses[i]   
        
        return subclass
    
    @property
    def weights(self):
        return self._w.value
    
    @property
    def alpha(self):
        return self._alpha.value[0]
    
    @property
    def risk(self):
        return self._risk.value
    
    @property
    def alpha_limit(self):
        return self._alpha_limit
    
    @property
    def risk_limit(self):
        return self._risk_limit
    
#%%

class _Interface(ABC):
    @abstractmethod
    def _define_prob(self):
        raise NotImplementedError
        
class _MaxAlpha(_Interface):
    
    def _define_prob(self):
        
        bm_wt = self.rtn.copy()
        bm_wt['wt'] = np.where(bm_wt.ticker == self.benchmark, 1, 0)
        self._risk_limit = bm_wt.wt @ self.cov.to_numpy() @ bm_wt.wt
        
        self._constraints.append(self._risk <= self._risk_limit)
        objective = cp.Maximize(self._alpha)
        prob = cp.Problem(objective, self._constraints)
        
        return prob, self

class _MinRisk(_Interface):
    
    def _define_prob(self):
        
        benchmark = self.rtn.ticker == self.benchmark
        self._alpha_limit = self.rtn.loc[benchmark, 'yhat'].values[0]
        self._constraints.append(self._alpha >= self._alpha_limit)
        
        objective = cp.Minimize(self._risk)
        prob = cp.Problem(objective, self._constraints)
        
        return prob, self
