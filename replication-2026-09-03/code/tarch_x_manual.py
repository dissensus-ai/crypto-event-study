"""
Manual TARCH-X Implementation for Cryptocurrency Event Study
==========================================================

This module implements TARCH models with exogenous variables in the variance equation,
following Engle & Ng (1993) and extending to include event dummies and sentiment variables.

Model Specification (GJR-GARCH Form):
σ²_t = ω + α₁ε²_{t-1} + γ₁ε²_{t-1}I(ε_{t-1}<0) + β₁σ²_{t-1} + Σδⱼx_{j,t}

Where:
- ω: intercept (omega) - baseline variance level
- α₁: ARCH effect (alpha) - response to recent squared shocks
- γ₁: leverage/asymmetry effect (gamma) - ADDITIONAL response to negative shocks
- β₁: GARCH effect (beta) - persistence of conditional variance
- δⱼ: coefficients on exogenous variables x_{j,t} (event dummies, sentiment)
- I(ε_{t-1}<0): indicator function for negative returns

LEVERAGE EFFECT INTERPRETATION:
-------------------------------
This implementation follows the GJR-GARCH specification (Glosten, Jagannathan, Runkle 1993):
- For positive shocks (ε_{t-1} > 0): volatility impact = α₁
- For negative shocks (ε_{t-1} < 0): volatility impact = α₁ + γ₁
- If γ₁ > 0: negative returns increase volatility MORE than positive returns (leverage effect)
- Total asymmetry = γ₁ (directly interpretable as additional volatility from bad news)

This differs from the original TARCH specification of Zakoian (1994) which uses:
σ_t (not σ²_t) with separate coefficients for positive and negative shocks.

Distribution: Student-t with degrees of freedom ν (captures fat tails in crypto returns)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t
from scipy.special import gamma
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class TARCHXResults:
    """Container for TARCH-X estimation results."""
    converged: bool
    params: Dict[str, float]
    std_errors: Dict[str, float]
    pvalues: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    volatility: pd.Series
    residuals: pd.Series
    event_effects: Dict[str, float]
    sentiment_effects: Dict[str, float]
    leverage_effect: float
    iterations: int
    
    def summary(self) -> str:
        """Generate summary statistics."""
        summary = f"""
TARCH-X Model Results
=====================
Converged: {self.converged}
Log-likelihood: {self.log_likelihood:.4f}
AIC: {self.aic:.4f}
BIC: {self.bic:.4f}

Variance Equation Parameters:
----------------------------
omega     = {self.params.get('omega', np.nan):.6f} ({self.pvalues.get('omega', np.nan):.4f})
alpha[1]  = {self.params.get('alpha', np.nan):.6f} ({self.pvalues.get('alpha', np.nan):.4f})
gamma[1]  = {self.params.get('gamma', np.nan):.6f} ({self.pvalues.get('gamma', np.nan):.4f})
beta[1]   = {self.params.get('beta', np.nan):.6f} ({self.pvalues.get('beta', np.nan):.4f})
nu        = {self.params.get('nu', np.nan):.6f} ({self.pvalues.get('nu', np.nan):.4f})

Event Effects:
--------------"""
        
        for event, coef in self.event_effects.items():
            pval = self.pvalues.get(event, np.nan)
            summary += f"\n{event:<20} = {coef:+.6f} ({pval:.4f})"
            
        if self.sentiment_effects:
            summary += "\n\nSentiment Effects:\n------------------"
            for sent, coef in self.sentiment_effects.items():
                pval = self.pvalues.get(sent, np.nan)
                summary += f"\n{sent:<20} = {coef:+.6f} ({pval:.4f})"
        
        return summary


class TARCHXEstimator:
    """
    Manual TARCH-X model estimator with exogenous variables in variance equation.
    """
    
    def __init__(self, returns: pd.Series, exog_vars: Optional[pd.DataFrame] = None):
        """
        Initialize TARCH-X estimator.
        
        Args:
            returns: Series of log returns (already multiplied by 100)
            exog_vars: DataFrame of exogenous variables for variance equation
        """
        self.returns = returns.dropna()
        
        if exog_vars is not None:
            # Align exogenous variables with returns
            self.exog_vars = exog_vars.loc[self.returns.index].fillna(0)
            self.has_exog = True
            self.n_exog = self.exog_vars.shape[1]
            self.exog_names = list(self.exog_vars.columns)
        else:
            self.exog_vars = None
            self.has_exog = False
            self.n_exog = 0
            self.exog_names = []
        
        self.n_obs = len(self.returns)
        self.param_names = ['omega', 'alpha', 'gamma', 'beta', 'nu'] + self.exog_names
        self.n_params = 5 + self.n_exog  # parameters optimized in the variance likelihood
        self.n_ic_params = self.n_params + 1  # also count the profiled sample mean in AIC/BIC
        
    def _unpack_params(self, params: np.ndarray) -> Dict[str, float]:
        """Unpack parameter vector into named dictionary."""
        param_dict = {
            'omega': params[0],
            'alpha': params[1], 
            'gamma': params[2],
            'beta': params[3],
            'nu': params[4]
        }
        
        # Add exogenous variable coefficients
        for i, name in enumerate(self.exog_names):
            param_dict[name] = params[5 + i]
            
        return param_dict
    
    def _variance_recursion(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute conditional variance and residuals using recursion.
        
        Args:
            params: Parameter vector [omega, alpha, gamma, beta, nu, delta1, delta2, ...]
            
        Returns:
            Tuple of (conditional_variance, residuals)
        """
        param_dict = self._unpack_params(params)
        omega = param_dict['omega']
        alpha = param_dict['alpha']
        gamma = param_dict['gamma'] 
        beta = param_dict['beta']
        
        # Initialize arrays
        variance = np.zeros(self.n_obs)
        # Demean returns to create proper residuals for GARCH estimation
        mean_return = self.returns.mean()
        residuals = (self.returns - mean_return).values
        
        # Initialize variance (unconditional variance estimate)
        variance[0] = np.var(self.returns)
        
        # Recursive computation
        for t in range(1, self.n_obs):
            # Previous squared residual
            eps_sq_prev = residuals[t-1] ** 2
            
            # Leverage term (negative return indicator)
            leverage_term = gamma * eps_sq_prev * (residuals[t-1] < 0)
            
            # Base TARCH terms
            variance[t] = (omega + 
                          alpha * eps_sq_prev + 
                          leverage_term + 
                          beta * variance[t-1])
            
            # Add exogenous variables if present
            if self.has_exog:
                for i, exog_name in enumerate(self.exog_names):
                    delta = param_dict[exog_name]
                    exog_value = self.exog_vars.iloc[t, i]
                    variance[t] += delta * exog_value
            
            # Ensure variance is positive
            variance[t] = max(variance[t], 1e-8)
        
        return variance, residuals
    
    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute negative log-likelihood for Student-t TARCH-X model.
        
        Args:
            params: Parameter vector
            
        Returns:
            Negative log-likelihood value
        """
        try:
            param_dict = self._unpack_params(params)
            nu = param_dict['nu']
            
            # Compute conditional variance
            variance, residuals = self._variance_recursion(params)
            
            # Standardized residuals
            std_residuals = residuals / np.sqrt(variance)
            
            # Student-t log-likelihood (excluding constants)
            # L(θ) = Σ[log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(π(ν-2)) - 0.5*log(σ²_t) 
            #        - ((ν+1)/2)*log(1 + ε²_t/(σ²_t*(ν-2)))]
            
            log_lik = 0
            for t in range(self.n_obs):
                # Log of gamma functions
                log_gamma_term = (np.log(gamma((nu + 1) / 2)) - 
                                 np.log(gamma(nu / 2)) - 
                                 0.5 * np.log(np.pi * (nu - 2)))
                
                # Variance term
                log_var_term = -0.5 * np.log(variance[t])
                
                # Density term
                density_term = -((nu + 1) / 2) * np.log(1 + std_residuals[t]**2 / (nu - 2))
                
                log_lik += log_gamma_term + log_var_term + density_term
            
            # Return negative log-likelihood for minimization
            return -log_lik
            
        except (ValueError, OverflowError, RuntimeWarning):
            # Return large positive value if computation fails
            return 1e8
    
    def _parameter_constraints(self) -> List[Dict]:
        """Define parameter constraints for optimization."""
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0] - 1e-8},  # omega > 0
            {'type': 'ineq', 'fun': lambda x: x[1] - 1e-8},  # alpha > 0  
            {'type': 'ineq', 'fun': lambda x: x[3] - 1e-8},  # beta > 0
            {'type': 'ineq', 'fun': lambda x: x[4] - 2.1},   # nu > 2 (for finite variance)
            {'type': 'ineq', 'fun': lambda x: 50 - x[4]},    # nu < 50 (for numerical stability)
            # Stationarity: alpha + beta + |gamma|/2 < 1
            {'type': 'ineq', 'fun': lambda x: 0.999 - (x[1] + x[3] + abs(x[2])/2)}
        ]
        return constraints
    
    def _get_starting_values(self) -> np.ndarray:
        """Generate reasonable starting values for optimization."""
        # Estimate initial variance
        sample_var = np.var(self.returns)
        
        # Starting values based on typical GARCH estimates
        start_vals = np.array([
            sample_var * 0.1,  # omega (small fraction of unconditional variance)
            0.05,              # alpha 
            0.05,              # gamma (leverage effect)
            0.85,              # beta (high persistence)
            5.0                # nu (moderate heavy tails)
        ])
        
        # Add zeros for exogenous variables (will be estimated)
        if self.has_exog:
            start_vals = np.append(start_vals, np.zeros(self.n_exog))
        
        return start_vals
    
    def estimate(self, method: str = 'SLSQP', max_iter: int = 1000) -> TARCHXResults:
        """
        Estimate TARCH-X model using maximum likelihood.
        
        Args:
            method: Optimization method ('SLSQP', 'L-BFGS-B', 'trust-constr')
            max_iter: Maximum number of iterations
            
        Returns:
            TARCHXResults object with estimation results
        """
        print(f"Estimating TARCH-X model with {self.n_exog} exogenous variables...")
        
        # Starting values
        start_vals = self._get_starting_values()
        
        # Parameter bounds (must be compatible with stationarity constraint)
        # Since alpha + beta + |gamma|/2 < 0.999, and max(|gamma|/2) = 0.25,
        # we need alpha + beta < 0.75 in worst case
        # Setting beta < 0.95 leaves room for alpha (typically 0.03-0.08)
        bounds = [
            (1e-8, None),      # omega > 0
            (1e-8, 0.3),       # 0 < alpha < 0.3
            (-0.5, 0.5),       # -0.5 < gamma < 0.5 (leverage can be negative)
            (1e-8, 0.95),      # 0 < beta < 0.95 (compatible with stationarity)
            (2.1, 50),         # 2 < nu < 50
        ]
        
        # Add bounds for exogenous variables (can be negative or positive)
        for _ in range(self.n_exog):
            bounds.append((None, None))  # Event/sentiment coefficients unbounded (Oct 28 fix)
        
        # Optimization
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                result = minimize(
                    fun=self._log_likelihood,
                    x0=start_vals,
                    method=method,
                    bounds=bounds,
                    constraints=self._parameter_constraints(),
                    options={'maxiter': max_iter, 'disp': False}
                )
            
            # Check convergence
            converged = result.success and result.fun < 1e6
            
            if not converged:
                print(f"  [WARNING] Optimization did not converge: {result.message}")
            
            # Extract results
            optimal_params = result.x
            param_dict = self._unpack_params(optimal_params)
            
            # Compute final variance and residuals
            variance, residuals = self._variance_recursion(optimal_params)
            volatility = pd.Series(np.sqrt(variance), index=self.returns.index)
            residuals_series = pd.Series(residuals, index=self.returns.index)
            
            # Compute standard errors from Hessian
            std_errors, pvalues = self._compute_standard_errors(optimal_params)
            
            # Information criteria
            log_lik = -result.fun
            aic = 2 * self.n_ic_params - 2 * log_lik
            bic = np.log(self.n_obs) * self.n_ic_params - 2 * log_lik
            
            # Separate event and sentiment effects
            event_effects = {}
            sentiment_effects = {}
            
            for name in self.exog_names:
                if any(event_word in name.lower() for event_word in ['event', 'infrastructure', 'regulatory']):
                    event_effects[name] = param_dict[name]
                elif any(sent_word in name.lower() for sent_word in ['sentiment', 'gdelt', 'tone']):
                    sentiment_effects[name] = param_dict[name]
                else:
                    event_effects[name] = param_dict[name]  # Default to event
            
            print(f"  [OK] Converged in {result.nit} iterations")
            print(f"  Log-likelihood: {log_lik:.2f}")
            print(f"  AIC: {aic:.2f}, BIC: {bic:.2f}")
            
            return TARCHXResults(
                converged=converged,
                params=param_dict,
                std_errors=std_errors,
                pvalues=pvalues,
                log_likelihood=log_lik,
                aic=aic,
                bic=bic,
                volatility=volatility,
                residuals=residuals_series,
                event_effects=event_effects,
                sentiment_effects=sentiment_effects,
                leverage_effect=param_dict['gamma'],
                iterations=result.nit
            )
            
        except Exception as e:
            print(f"  [FAIL] Estimation failed: {str(e)}")
            
            # Return failed result
            return TARCHXResults(
                converged=False,
                params={},
                std_errors={},
                pvalues={},
                log_likelihood=np.nan,
                aic=np.nan,
                bic=np.nan,
                volatility=pd.Series(),
                residuals=pd.Series(),
                event_effects={},
                sentiment_effects={},
                leverage_effect=np.nan,
                iterations=0
            )
    
    def _compute_standard_errors(self, params: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute standard errors using numerical Hessian.
        
        Args:
            params: Optimal parameter vector
            
        Returns:
            Tuple of (standard_errors_dict, pvalues_dict)
        """
        try:
            # Numerical Hessian computation
            hessian = self._numerical_hessian(params)
            
            # Covariance matrix (inverse of Hessian)
            cov_matrix = np.linalg.inv(hessian)
            
            # A non-positive covariance diagonal means the local numerical
            # Hessian does not support a real standard error for that parameter.
            # Preserve that fact as NaN without emitting a misleading sqrt
            # runtime warning; these Hessian SEs are diagnostic only.
            cov_diag = np.diag(cov_matrix)
            std_errs = np.full(cov_diag.shape, np.nan, dtype=float)
            valid_diag = np.isfinite(cov_diag) & (cov_diag > 0)
            std_errs[valid_diag] = np.sqrt(cov_diag[valid_diag])
            
            # Compute t-statistics and p-values
            with np.errstate(divide="ignore", invalid="ignore"):
                t_stats = params / std_errs

            # Use Student-t distribution with n-k degrees of freedom
            # CRITICAL: Validate sufficient degrees of freedom
            dof = self.n_obs - self.n_params
            if dof <= 0:
                print(f"  [ERROR] Insufficient degrees of freedom: n_obs={self.n_obs}, n_params={self.n_params}")
                print(f"          DOF = {dof} <= 0 - Cannot compute valid standard errors")
                # Return NaN values - model is over-parameterized
                std_errors = {name: np.nan for name in self.param_names}
                pvalues = {name: np.nan for name in self.param_names}
                return std_errors, pvalues

            pvals = 2 * (1 - student_t.cdf(np.abs(t_stats), dof))

            # Create dictionaries
            std_errors = dict(zip(self.param_names, std_errs))
            pvalues = dict(zip(self.param_names, pvals))

            return std_errors, pvalues
            
        except (np.linalg.LinAlgError, ValueError):
            print("  [WARNING] Could not compute standard errors")
            
            # Return NaN values
            std_errors = {name: np.nan for name in self.param_names}
            pvalues = {name: np.nan for name in self.param_names}
            
            return std_errors, pvalues
    
    def _numerical_hessian(self, params: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Compute numerical Hessian matrix using central differences.
        
        Args:
            params: Parameter vector
            h: Step size for numerical differentiation
            
        Returns:
            Hessian matrix
        """
        n = len(params)
        hessian = np.zeros((n, n))
        
        # Central difference approximation for Hessian
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal elements: second derivative
                    params_plus = params.copy()
                    params_minus = params.copy()
                    params_plus[i] += h
                    params_minus[i] -= h
                    
                    f_plus = self._log_likelihood(params_plus)
                    f_minus = self._log_likelihood(params_minus)
                    f_center = self._log_likelihood(params)
                    
                    hessian[i, j] = (f_plus - 2*f_center + f_minus) / (h**2)
                else:
                    # Off-diagonal elements: mixed partial derivatives
                    params_pp = params.copy()
                    params_pm = params.copy() 
                    params_mp = params.copy()
                    params_mm = params.copy()
                    
                    params_pp[i] += h
                    params_pp[j] += h
                    
                    params_pm[i] += h
                    params_pm[j] -= h
                    
                    params_mp[i] -= h
                    params_mp[j] += h
                    
                    params_mm[i] -= h
                    params_mm[j] -= h
                    
                    f_pp = self._log_likelihood(params_pp)
                    f_pm = self._log_likelihood(params_pm)
                    f_mp = self._log_likelihood(params_mp)
                    f_mm = self._log_likelihood(params_mm)
                    
                    hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h**2)
        
        return hessian


def estimate_tarch_x_manual(returns: pd.Series, 
                           exog_vars: Optional[pd.DataFrame] = None,
                           method: str = 'SLSQP') -> TARCHXResults:
    """
    Convenience function to estimate TARCH-X model.
    
    Args:
        returns: Series of log returns (should be in percentage terms)
        exog_vars: DataFrame of exogenous variables
        method: Optimization method
        
    Returns:
        TARCHXResults object
    """
    estimator = TARCHXEstimator(returns, exog_vars)
    return estimator.estimate(method=method)


# Example usage for testing
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    n_obs = 1000
    
    # Synthetic returns with GARCH properties
    returns = np.random.normal(0, 1, n_obs)
    for i in range(1, n_obs):
        returns[i] = returns[i] * np.sqrt(0.01 + 0.05 * returns[i-1]**2 + 0.9 * 0.01)
    
    returns = pd.Series(returns * 100, index=pd.date_range('2020-01-01', periods=n_obs))
    
    # Synthetic event dummies
    event_dummy = np.zeros(n_obs)
    event_dummy[100:107] = 1  # 7-day event window
    event_dummy[500:507] = 1  # Another event
    
    exog_df = pd.DataFrame({
        'D_infrastructure': event_dummy
    }, index=returns.index)
    
    # Estimate model
    results = estimate_tarch_x_manual(returns, exog_df)
    print(results.summary())
