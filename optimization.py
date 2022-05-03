# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 00:53:01 2022

Class for optimizing portfolio weights

@author: tommy
"""

#%% libs

import logging
import cvxpy as cp
import numpy as np
import pandas as pd
from .utils import todate
from pydantic import BaseModel

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
    lg : logging.logger
        logger used for warnings
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
    '''
    
    rtn: pd.core.frame.DataFrame 
    cov: pd.core.frame.DataFrame 
    lg: logging.Logger
    benchmark = 'SPY' 
    opt_kind = 'max_alpha'
    bm_min = .8 
    max_wt = .05  
    max_cash_wt = .07
    min_nonzero_wt = .01 
    solver = 'XPRESS' 
    verbose = False
    sol_meta = {} #TODO: use _sol_meta
    _handlers = {} #TODO clean up function factory

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
        self.rtn, self.cov = self._conform_opt_inputs()
        prob = self._define_prob()
        prob.solve(verbose = self.verbose, solver = self.solver)
        sol_meta = self.sol_meta.copy()
        sol_meta['exposure'] = sol_meta['exposure'].value[0]
        sol_meta['risk'] = sol_meta['risk'].value
        opt = self.rtn.copy()
        opt['wt'] = prob.variables()[0].value
        opt = opt.sort_values('yhat')
        return opt, sol_meta

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

    def _define_prob(self, _handlers = _handlers):
        '''
        Defines and returns a cvxpy Problem instance
        '''
        
        r = self.rtn.copy()
        alpha = self.rtn.yhat.to_numpy().reshape([len(r), 1])
        sigma = self.cov.to_numpy()
            
        w, b = cp.Variable(len(r)), cp.Variable(len(r), boolean=True)  
        constraints = self._base_constraints(w, b)
        
        risk_limit, exp_limit = None, None
        exposure, risk = alpha.T @ w, cp.quad_form(w, sigma)
                
        objective, constraints = self._get_opt_kind(sigma, exposure, risk, 
                                                    constraints, _handlers)
        
        prob = cp.Problem(objective, constraints)
        
        self.sol_meta = {'exposure': exposure, 'risk': risk,
                          'exp_limit': exp_limit, 'risk_limit': risk_limit}

        return prob

    def _base_constraints(self, w, b):
        '''
        Constraints that are commom to the different kinds of optimizations
    
        Parameters
        ----------
        w : cvxpy.Variable
            weights to be optimized.
        b : cvxpy.Variable
            binary variable used to constrain non-zero weights to be greater 
            than a specified amount. cvxpy doesn't support semi-continuous vars, 
            and this approach for defining the constraint makes this a MIQP
    
        Returns
        -------
        constraints : list
            list of the base constraints.
    
        '''
        
        r = self.rtn
        n = w.shape[0]
        min_wts = np.where(r.ticker == self.benchmark, self.bm_min, 0)
        max_wts = np.where(r.ticker == self.benchmark, 1, self.max_wt)
        max_wts = np.where(r.ticker == 'cash', self.max_cash_wt, max_wts)
        
        constraints = [cp.sum(w) == 1,
                       w >= min_wts,
                       w <= max_wts,
                       w >= cp.multiply(b, np.ones(n) * self.min_nonzero_wt),
                       w <= cp.multiply(b, np.ones(n))
                       ]
        
        return constraints

    def _register_handler(kind, _handlers = _handlers):
        def _wrapper(fn):
            _handlers[kind] = fn
            return fn
        return _wrapper
    
    def _get_opt_kind(self, sigma, exposure, risk, constraints, _handlers):
        rtn, bmrk, kind = self.rtn, self.benchmark, self.opt_kind
        try:
            return _handlers[kind](rtn, sigma, exposure, risk, bmrk, 
                                   constraints)
        except KeyError:
            raise NotImplementedError(f"opt_kind ='{kind}' is not supported")
    
    @_register_handler('max_alpha')
    def max_alpha(rtn, sigma, exposure, risk, bmrk, constraints):
        bm_wt = rtn.copy()
        bm_wt['wt'] = np.where(bm_wt.ticker == bmrk, 1, 0)
        risk_limit = bm_wt.wt @ sigma @ bm_wt.wt
        constraints.append(risk <= risk_limit)
        objective = cp.Maximize(exposure)
        return objective, constraints
        
    @_register_handler('min_risk')
    def min_risk(rtn, sigma, exposure, risk, bmrk, constraints):   
        r = rtn.copy()
        exp_limit = r.loc[r.ticker == bmrk, 'yhat'].values[0]
        constraints.append(exposure >= exp_limit)
        objective = cp.Minimize(risk)
        return objective, constraints  
