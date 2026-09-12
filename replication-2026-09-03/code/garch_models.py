"""
GARCH model estimation module for cryptocurrency event study.
Implements GARCH(1,1), TARCH(1,1), and TARCH-X models with robust estimation.
"""

import pandas as pd
import numpy as np

# arch package is optional - only needed for baseline GARCH/TARCH models
# Manual TARCH-X implementation (tarch_x_manual.py) works without arch
try:
    from arch import arch_model
    from arch.univariate import GARCH, EGARCH, ConstantMean, StudentsT, Normal
    try:
        from arch.univariate import GJRGARCH as GJR
    except ImportError:
        GJR = None
    ARCH_AVAILABLE = True
except ImportError:
    # arch package not installed - will use manual implementation only
    arch_model = None
    GARCH = EGARCH = ConstantMean = StudentsT = Normal = GJR = None
    ARCH_AVAILABLE = False
from typing import Dict, List, Optional, Tuple, Any
import warnings
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from data_preparation import DataPreparation
from tarch_x_manual import estimate_tarch_x_manual


@dataclass
class ModelResults:
    """Container for model estimation results."""
    model_type: str
    crypto: str
    aic: float
    bic: float
    log_likelihood: float
    parameters: Dict[str, float]
    std_errors: Dict[str, float]
    pvalues: Dict[str, float]
    convergence: bool
    iterations: int
    volatility: pd.Series
    residuals: pd.Series
    leverage_effect: Optional[float] = None
    event_effects: Optional[Dict[str, float]] = None
    sentiment_effects: Optional[Dict[str, float]] = None
    event_std_errors: Optional[Dict[str, float]] = None
    event_pvalues: Optional[Dict[str, float]] = None


class GARCHModels:
    """
    Main class for GARCH model estimation and analysis.
    Implements the three-model progression with fallback strategies.
    """

    def __init__(self, data: pd.DataFrame, crypto: str):
        """
        Initialize GARCH model estimator.

        Args:
            data: Prepared DataFrame with returns and exogenous variables
            crypto: Cryptocurrency symbol for identification
        """
        self.data = data.copy()
        self.crypto = crypto
        self.returns = data['returns_winsorized'].dropna()

        # Store exogenous variables if present
        self.has_events = any(col.startswith('D_') for col in data.columns)
        self.has_sentiment = any('gdelt' in col or 'reg_decomposed' in col or 'infra_decomposed' in col
                                for col in data.columns)

        # Model specifications
        self.mean_model = 'Constant'  # Can be 'Constant', 'Zero', or 'AR'
        self.vol_models = {
            'GARCH': GARCH,
            'TARCH': GJR,  # GJR-GARCH is equivalent to TARCH
            'EGARCH': EGARCH
        }

        # Estimation options
        self.disp = 'off'  # Turn off optimization display
        self.options = {'maxiter': 1000}

        # Results storage
        self.results = {}

        # Track fallback attempts for debugging
        self._is_fallback_attempt = False

    def estimate_garch_11(self) -> ModelResults:
        """
        Estimate GARCH(1,1) baseline model.
        σ²_t = ω + α₁ε²_{t-1} + β₁σ²_{t-1}

        Returns:
            ModelResults object with estimation results
        """
        print(f"Estimating GARCH(1,1) for {self.crypto}...")

        if not ARCH_AVAILABLE:
            print("  [SKIP] arch package not available - install with: pip install arch")
            return self._create_failed_result('GARCH(1,1)')

        try:
            # Create GARCH(1,1) model with Student's t distribution
            model = arch_model(
                self.returns,
                mean=self.mean_model,
                vol='GARCH',
                p=1,
                q=1,
                dist='StudentsT'
            )

            # Estimate with robust standard errors
            res = model.fit(
                disp=self.disp,
                options=self.options,
                cov_type='robust'
            )

            # Extract results
            results = self._extract_results(res, 'GARCH(1,1)')
            self.results['GARCH'] = results

            print(f"  [OK] GARCH(1,1) converged in {results.iterations} iterations")
            print(f"  AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")

            return results

        except Exception as e:
            print(f"  [FAIL] GARCH(1,1) estimation failed: {str(e)}")
            return self._create_failed_result('GARCH(1,1)')

    def estimate_tarch_11(self) -> ModelResults:
        """
        Estimate TARCH(1,1) model with asymmetric effects.
        σ²_t = ω + α₁ε²_{t-1} + γ₁ε²_{t-1}I(ε_{t-1}<0) + β₁σ²_{t-1}

        Returns:
            ModelResults object with estimation results
        """
        print(f"Estimating TARCH(1,1) for {self.crypto}...")

        if not ARCH_AVAILABLE:
            print("  [SKIP] arch package not available - install with: pip install arch")
            return self._create_failed_result('TARCH(1,1)')

        try:
            # Create TARCH(1,1) model (GJR-GARCH) with Student's t
            model = arch_model(
                self.returns,
                mean=self.mean_model,
                vol='GARCH',
                p=1,
                o=1,  # This makes it GJR/TARCH
                q=1,
                dist='StudentsT'
            )

            # Estimate with robust standard errors
            res = model.fit(
                disp=self.disp,
                options=self.options,
                cov_type='robust'
            )

            # Extract results
            results = self._extract_results(res, 'TARCH(1,1)')

            # Extract leverage effect (gamma parameter)
            if 'gamma[1]' in res.params:
                results.leverage_effect = res.params['gamma[1]']

            self.results['TARCH'] = results

            print(f"  [OK] TARCH(1,1) converged in {results.iterations} iterations")
            print(f"  AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")
            if results.leverage_effect:
                print(f"  Leverage effect (gamma): {results.leverage_effect:.4f}")

            return results

        except Exception as e:
            print(f"  [FAIL] TARCH(1,1) estimation failed: {str(e)}")
            return self._create_failed_result('TARCH(1,1)')


    


    def estimate_tarch_x(self, use_individual_events: bool = True,
                        include_sentiment: bool = True) -> ModelResults:
        """Estimate TARCH-X model with exogenous variables in the variance equation."""
        print(f"Estimating TARCH-X for {self.crypto}...")

        exog_vars = self._prepare_exogenous_variables(
            use_individual_events,
            include_sentiment
        )

        if exog_vars is None or exog_vars.empty:
            print("  [WARNING] No exogenous variables available for TARCH-X")
            return self._create_failed_result('TARCH-X')

        try:
            # Align exogenous variables with returns
            exog_aligned = exog_vars.loc[self.returns.index].fillna(0)

            # MULTICOLLINEARITY CHECK - Critical for valid inference
            if len(exog_aligned.columns) > 1:
                corr_matrix = exog_aligned.corr().abs()
                # Check for high correlations (>0.95) excluding diagonal
                np.fill_diagonal(corr_matrix.values, 0)
                max_corr = corr_matrix.max().max()
                if max_corr > 0.95:
                    print(f"  [WARNING] High multicollinearity detected! Max correlation: {max_corr:.3f}")
                    print(f"            Standard errors may be inflated and unreliable")
                    # Find which variables are highly correlated
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if corr_matrix.iloc[i, j] > 0.95:
                                high_corr_pairs.append(
                                    f"{corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i,j]:.3f}"
                                )
                    if high_corr_pairs:
                        print(f"            High correlations found:")
                        for pair in high_corr_pairs[:5]:  # Show max 5 pairs
                            print(f"              {pair}")

            # Use manual TARCH-X implementation for proper variance equation specification
            print(f"  Using manual TARCH-X implementation with {len(exog_aligned.columns)} exogenous variables")

            manual_results = estimate_tarch_x_manual(
                returns=self.returns,
                exog_vars=exog_aligned,
                method='SLSQP'
            )

            if not manual_results.converged:
                print("  [FAIL] Manual TARCH-X did not converge")
                return self._create_failed_result('TARCH-X')

            # Convert manual results to ModelResults format for compatibility
            results = ModelResults(
                model_type='TARCH-X',
                crypto=self.crypto,
                aic=manual_results.aic,
                bic=manual_results.bic,
                log_likelihood=manual_results.log_likelihood,
                parameters=manual_results.params,
                std_errors=manual_results.std_errors,
                pvalues=manual_results.pvalues,
                convergence=manual_results.converged,
                iterations=manual_results.iterations,
                volatility=manual_results.volatility,
                residuals=manual_results.residuals,
                leverage_effect=manual_results.leverage_effect,
                event_effects=manual_results.event_effects,
                sentiment_effects=manual_results.sentiment_effects,
                event_std_errors={k: manual_results.std_errors.get(k, np.nan)
                                 for k in manual_results.event_effects.keys()},
                event_pvalues={k: manual_results.pvalues.get(k, np.nan)
                              for k in manual_results.event_effects.keys()}
            )

            # Display key results during estimation
            print(f"  [OK] Manual TARCH-X converged in {manual_results.iterations} iterations")
            print(f"  Log-likelihood: {manual_results.log_likelihood:.2f}")
            print(f"  AIC: {manual_results.aic:.2f}, BIC: {manual_results.bic:.2f}")
            print(f"  Event coefficients: {len(manual_results.event_effects)}")
            if manual_results.sentiment_effects:
                print(f"  Sentiment coefficients: {len(manual_results.sentiment_effects)}")

            # Display key event effects
            if manual_results.event_effects:
                print("  Event variance coefficients:")
                for name, coef in manual_results.event_effects.items():
                    p_val = manual_results.pvalues.get(name, np.nan)
                    sig_stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                    p_display = f", p={p_val:.3f}" if not np.isnan(p_val) else ""
                    print(f"    {name}: {coef:+.4f}{sig_stars}{p_display}")

            # Display sentiment effects if present
            if manual_results.sentiment_effects:
                print("  Sentiment variance coefficients:")
                for name, coef in manual_results.sentiment_effects.items():
                    p_val = manual_results.pvalues.get(name, np.nan)
                    sig_stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                    p_display = f", p={p_val:.3f}" if not np.isnan(p_val) else ""
                    print(f"    {name}: {coef:+.4f}{sig_stars}{p_display}")

            # Display leverage effect
            if manual_results.leverage_effect and not np.isnan(manual_results.leverage_effect):
                gamma_pval = manual_results.pvalues.get('gamma', np.nan)
                sig_stars = "***" if gamma_pval < 0.01 else "**" if gamma_pval < 0.05 else "*" if gamma_pval < 0.10 else ""
                print(f"  Leverage effect (gamma): {manual_results.leverage_effect:.4f}{sig_stars}")

            self.results['TARCH-X'] = results
            return results

        except Exception as e:
            print(f"  [FAIL] TARCH-X estimation failed: {str(e)}")

            if use_individual_events and not self._is_fallback_attempt:
                print("  -> Attempting fallback with aggregated event type dummies...")
                self._is_fallback_attempt = True
                return self.estimate_tarch_x(
                    use_individual_events=False,
                    include_sentiment=include_sentiment
                )

            return self._create_failed_result('TARCH-X')

    def estimate_all_models(self) -> Dict[str, ModelResults]:
        """
        Estimate all three models in sequence.

        Returns:
            Dictionary of model results
        """
        print(f"\n{'='*60}")
        print(f"Estimating GARCH models for {self.crypto}")
        print(f"{'='*60}")

        # Reset fallback flag
        self._is_fallback_attempt = False

        # Estimate models in sequence
        garch_results = self.estimate_garch_11()
        tarch_results = self.estimate_tarch_11()
        tarchx_results = self.estimate_tarch_x(use_individual_events=False)

        # Store all results
        all_results = {
            'GARCH(1,1)': garch_results,
            'TARCH(1,1)': tarch_results,
            'TARCH-X': tarchx_results
        }

        # Model comparison
        self._compare_models(all_results)

        return all_results

    def _prepare_exogenous_variables(self, use_individual_events: bool,
                                    include_sentiment: bool) -> Optional[pd.DataFrame]:
        """
        Prepare exogenous variables for TARCH-X model.

        Args:
            use_individual_events: Whether to use individual event dummies
            include_sentiment: Whether to include sentiment variables

        Returns:
            DataFrame with exogenous variables or None
        """
        exog_vars = []

        # Add event dummies
        if self.has_events:
            if use_individual_events:
                event_cols = [col for col in self.data.columns if col.startswith('D_')]
                if event_cols:
                    exog_vars.extend(sorted(event_cols))
            else:
                if 'D_infrastructure' in self.data.columns:
                    exog_vars.append('D_infrastructure')
                if 'D_regulatory' in self.data.columns:
                    exog_vars.append('D_regulatory')

        # Add sentiment variables
        if include_sentiment and self.has_sentiment:
            sentiment_cols = [col for col in self.data.columns
                            if 'gdelt_normalized' in col or
                               'reg_decomposed' in col or
                               'infra_decomposed' in col]
            exog_vars.extend(sentiment_cols)

        if not exog_vars:
            return None

        # Handle special overlap adjustments
        exog_df = self.data[exog_vars].copy()

        # The overlap adjustments are already in the data from data_preparation
        # Just ensure we handle the 0.5 values correctly

        return exog_df

    def _extract_results(self, arch_result, model_type: str) -> ModelResults:
        """
        Extract results from arch model estimation with safer attribute access.

        Args:
            arch_result: Result object from arch package
            model_type: String identifier for model type

        Returns:
            ModelResults object
        """
        # Safer convergence checking with multiple fallbacks
        converged = bool(getattr(arch_result, 'converged',
                               getattr(arch_result, 'convergence_flag', 1) == 0))

        # Safer extraction of standard errors
        std_errors = {}
        if hasattr(arch_result, 'std_err'):
            std_errors = dict(arch_result.std_err)
        elif hasattr(arch_result, 'bse'):
            std_errors = dict(arch_result.bse)
        elif hasattr(arch_result, 'std_errors'):
            std_errors = dict(arch_result.std_errors)

        return ModelResults(
            model_type=model_type,
            crypto=self.crypto,
            aic=getattr(arch_result, 'aic', np.nan),
            bic=getattr(arch_result, 'bic', np.nan),
            log_likelihood=getattr(arch_result, 'loglikelihood', np.nan),
            parameters=dict(getattr(arch_result, 'params', {})),
            std_errors=std_errors,
            pvalues=dict(getattr(arch_result, 'pvalues', {})),
            convergence=converged,
            iterations=getattr(arch_result, 'iterations',
                             getattr(arch_result, 'num_params', 0)),
            volatility=getattr(arch_result, 'conditional_volatility',
                             pd.Series(dtype=float)),
            residuals=getattr(arch_result, 'resid',
                            pd.Series(dtype=float))
        )

    def _create_failed_result(self, model_type: str) -> ModelResults:
        """
        Create a failed result object for models that didn't converge.

        Args:
            model_type: String identifier for model type

        Returns:
            ModelResults object with NaN values
        """
        return ModelResults(
            model_type=model_type,
            crypto=self.crypto,
            aic=np.nan,
            bic=np.nan,
            log_likelihood=np.nan,
            parameters={},
            std_errors={},
            pvalues={},
            convergence=False,
            iterations=0,
            volatility=pd.Series(),
            residuals=pd.Series()
        )

    def _compare_models(self, results: Dict[str, ModelResults]):
        """
        Compare models using AIC and BIC.

        Args:
            results: Dictionary of model results
        """
        print(f"\n{'='*60}")
        print(f"Model Comparison for {self.crypto}")
        print(f"{'='*60}")

        comparison_data = []
        for model_name, result in results.items():
            if result.convergence:
                comparison_data.append({
                    'Model': model_name,
                    'AIC': result.aic,
                    'BIC': result.bic,
                    'Log-Likelihood': result.log_likelihood,
                    'Converged': result.convergence
                })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('AIC')

            print("\nModel Fit Statistics:")
            print(comparison_df.to_string(index=False))

            # Identify best model
            best_aic = comparison_df.iloc[0]['Model']
            best_bic = comparison_df.sort_values('BIC').iloc[0]['Model']

            print(f"\nBest model by AIC: {best_aic}")
            print(f"Best model by BIC: {best_bic}")
        else:
            print("No models converged successfully")

    def run_diagnostics(self, model_result: ModelResults) -> Dict[str, Any]:
        """
        Run diagnostic tests on model residuals.

        Args:
            model_result: ModelResults object to test

        Returns:
            Dictionary with diagnostic test results
        """
        if not model_result.convergence:
            return {'error': 'Model did not converge'}

        from scipy import stats

        residuals = model_result.residuals
        std_residuals = residuals / model_result.volatility

        diagnostics = {}

        # Ljung-Box test for autocorrelation
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(std_residuals.dropna(), lags=10, return_df=True)
        diagnostics['ljung_box'] = {
            'statistic': lb_test['lb_stat'].iloc[-1],
            'pvalue': lb_test['lb_pvalue'].iloc[-1]
        }

        # ARCH-LM test for remaining heteroskedasticity
        from statsmodels.stats.diagnostic import het_arch
        arch_test = het_arch(std_residuals.dropna(), nlags=5)
        diagnostics['arch_lm'] = {
            'statistic': arch_test[0],
            'pvalue': arch_test[1]
        }

        # Jarque-Bera test for normality
        jb_test = stats.jarque_bera(std_residuals.dropna())
        diagnostics['jarque_bera'] = {
            'statistic': jb_test[0],
            'pvalue': jb_test[1]
        }

        return diagnostics

    def extract_event_impacts(self) -> pd.DataFrame:
        """
        Extract and organize event impact coefficients from TARCH-X model.

        Returns:
            DataFrame with event impacts and significance
        """
        if 'TARCH-X' not in self.results or not self.results['TARCH-X'].convergence:
            print(f"No converged TARCH-X model for {self.crypto}")
            return pd.DataFrame()

        tarchx = self.results['TARCH-X']

        if not tarchx.event_effects:
            print(f"No event effects found in TARCH-X model for {self.crypto}")
            return pd.DataFrame()

        # Create DataFrame with event effects
        effects_data = []
        for event_var, coefficient in tarchx.event_effects.items():
            effects_data.append({
                'crypto': self.crypto,
                'event_variable': event_var,
                'coefficient': coefficient,
                'std_error': tarchx.std_errors.get(event_var, np.nan),
                'p_value': tarchx.pvalues.get(event_var, np.nan),
                'significant_5pct': tarchx.pvalues.get(event_var, 1.0) < 0.05,
                'significant_10pct': tarchx.pvalues.get(event_var, 1.0) < 0.10
            })

        return pd.DataFrame(effects_data)


def estimate_models_for_crypto(crypto: str, data: pd.DataFrame) -> Dict[str, ModelResults]:
    """
    Convenience function to estimate all models for a single cryptocurrency.

    Args:
        crypto: Cryptocurrency symbol
        data: Prepared DataFrame

    Returns:
        Dictionary of model results
    """
    modeler = GARCHModels(data, crypto)
    return modeler.estimate_all_models()


def estimate_models_for_all_cryptos(crypto_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, ModelResults]]:
    """
    Estimate models for all cryptocurrencies.

    Args:
        crypto_data: Dictionary mapping crypto symbols to DataFrames

    Returns:
        Nested dictionary of results by crypto and model
    """
    all_results = {}

    for crypto, data in crypto_data.items():
        print(f"\n{'='*80}")
        print(f"Processing {crypto.upper()}")
        print(f"{'='*80}")

        try:
            all_results[crypto] = estimate_models_for_crypto(crypto, data)
        except Exception as e:
            print(f"Error processing {crypto}: {str(e)}")
            continue

    return all_results