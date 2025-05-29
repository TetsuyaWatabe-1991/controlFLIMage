import numpy as np
import scipy.optimize as opt
from scipy.special import erfc
from scipy.linalg import inv, pinv
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
import warnings

class FLIMLifetimeFitter:
    """
    Fluorescence Lifetime Imaging (FLIM) data fitting using exponential-Gaussian convolution models.
    
    This implementation is based on the C# FLIMage software algorithms, supporting both 
    single and double exponential decay models convolved with Gaussian instrument response.
    """
    
    def __init__(self):
        self.max_iter = 100
        self.beta_tol = 1e-6
        self.r_tol = 1e-8
        self.tau_g_max = 500  # picoseconds
        self.tau_g_min = 60   # picoseconds
        self.max_tau = 10000  # picoseconds
        self.min_tau = 100    # picoseconds
        
    def exp_gauss_single(self, beta: np.ndarray, x: np.ndarray, pulse_interval: float) -> np.ndarray:
        """
        Single exponential decay convolved with Gaussian instrument response.
        
        Parameters:
        beta: [amplitude, decay_rate, tau_g, t0] where:
            - amplitude: peak amplitude 
            - decay_rate: 1/lifetime (in units^-1)
            - tau_g: Gaussian width of instrument response
            - t0: time offset
        x: time points
        pulse_interval: laser pulse interval (for periodic excitation)
        
        Returns:
        y: model values at time points x
        """
        amplitude, decay_rate, tau_g, t0 = beta
        
        # Main pulse contribution
        y1 = amplitude * np.exp(tau_g**2 * decay_rate**2 / 2 - (x - t0) * decay_rate)
        y2 = erfc((tau_g**2 * decay_rate - (x - t0)) / (np.sqrt(2) * tau_g))
        y_main = y1 * y2
        
        # Previous pulse contribution (for periodic excitation)
        y1_prev = amplitude * np.exp(tau_g**2 * decay_rate**2 / 2 - (x - t0 + pulse_interval) * decay_rate)
        y2_prev = erfc((tau_g**2 * decay_rate - (x - t0 + pulse_interval)) / (np.sqrt(2) * tau_g))
        y_prev = y1_prev * y2_prev
        
        return (y_main + y_prev) / 2
    
    def exp_gauss_double(self, beta: np.ndarray, x: np.ndarray, pulse_interval: float) -> np.ndarray:
        """
        Double exponential decay convolved with Gaussian instrument response.
        
        Parameters:
        beta: [amp1, rate1, amp2, rate2, tau_g, t0] where:
            - amp1, amp2: amplitudes of two components
            - rate1, rate2: decay rates (1/lifetime) of two components  
            - tau_g: Gaussian width of instrument response
            - t0: time offset
        x: time points
        pulse_interval: laser pulse interval
        
        Returns:
        y: model values at time points x
        """
        amp1, rate1, amp2, rate2, tau_g, t0 = beta
        
        # First component
        beta1 = np.array([amp1, rate1, tau_g, t0])
        y1 = self.exp_gauss_single(beta1, x, pulse_interval)
        
        # Second component  
        beta2 = np.array([amp2, rate2, tau_g, t0])
        y2 = self.exp_gauss_single(beta2, x, pulse_interval)
        
        return y1 + y2
    
    def calculate_jacobian(self, beta: np.ndarray, x: np.ndarray, model_func, 
                          fixed_params: list, *args) -> np.ndarray:
        """Calculate Jacobian matrix using finite differences, with support for fixed parameters."""
        n = len(x)
        p = len(beta)
        delta_step = self.beta_tol * 0.01
        
        jacobian = np.zeros((n, p))
        y_fit = model_func(beta, x, *args)
        
        for i in range(p):
            if i in fixed_params:
                # For fixed parameters, derivative is zero
                jacobian[:, i] = 0.0
                continue
                
            beta_new = beta.copy()
            delta = delta_step * beta[i] if beta[i] != 0 else delta_step * np.linalg.norm(beta)
            if delta == 0:
                delta = delta_step
            
            beta_new[i] += delta
            y_plus = model_func(beta_new, x, *args)
            jacobian[:, i] = (y_plus - y_fit) / delta
            
        return jacobian
    
    def levenberg_marquardt_step(self, jacobian: np.ndarray, residuals: np.ndarray, 
                                 lambda_param: float) -> np.ndarray:
        """Calculate Levenberg-Marquardt step."""
        jt = jacobian.T
        jtj = jt @ jacobian
        
        # Add damping term
        diag_indices = np.diag_indices_from(jtj)
        jtj[diag_indices] *= (1 + lambda_param)
        
        jtr = jt @ residuals
        
        try:
            step = inv(jtj) @ jtr
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            step = pinv(jtj) @ jtr
            
        return step
    
    def fit_single_exponential(self, x: np.ndarray, y: np.ndarray, 
                              ps_per_unit: float, sync_rate: float,
                              beta0: Optional[np.ndarray] = None,
                              weights: Optional[np.ndarray] = None,
                              fix_tau: Optional[float] = None) -> Dict[str, Any]:
        """
        Fit single exponential decay model.
        
        Parameters:
        x: time points (in time units)
        y: photon counts
        ps_per_unit: picoseconds per time unit
        sync_rate: laser sync rate (Hz)
        beta0: initial parameter guess [amplitude, decay_rate, tau_g, t0]
        weights: fitting weights (if None, Poisson weights are used)
        fix_tau: if provided, fix the lifetime to this value (in nanoseconds)
        
        Returns:
        Dictionary with fitting results
        """
        if weights is None:
            weights = np.where(y > 1, 1.0 / np.sqrt(y), 1.0)  # Original method - back to this
        
        # Poisson weights using expected counts (commented out - didn't improve much)
        # if weights is None:
        #     weights = None  # Will be calculated using fitted model
        
        # Calculate pulse interval
        pulse_interval = 1e12 / sync_rate / ps_per_unit
        
        # Initialize parameters if not provided
        if beta0 is None:
            max_val = np.max(y)
            max_idx = np.argmax(y)
            
            # Better estimate of mean lifetime from data
            weighted_time = np.sum(x * y) / np.sum(y) if np.sum(y) > 0 else len(x) / 2
            est_lifetime_bins = weighted_time - max_idx
            est_lifetime_bins = max(est_lifetime_bins, 10)  # Minimum reasonable lifetime
            
            tau_g = 100 / ps_per_unit  # Initial guess for instrument response
            
            if fix_tau is not None:
                # Convert fixed tau from ns to decay rate
                fixed_decay_rate = ps_per_unit / (fix_tau * 1000)
            else:
                fixed_decay_rate = 1 / est_lifetime_bins
            
            beta0 = np.array([
                max_val,                                     # amplitude
                fixed_decay_rate,                            # decay rate
                tau_g,                                       # instrument response width
                max_idx - tau_g                              # time offset
            ])
        
        # Parameter bounds
        if fix_tau is not None:
            # Fix the decay rate if tau is specified
            fixed_decay_rate = ps_per_unit / (fix_tau * 1000)
            bounds = [
                (0, np.inf),                              # amplitude > 0
                (fixed_decay_rate, fixed_decay_rate),     # decay rate fixed
                (self.tau_g_min / ps_per_unit, self.tau_g_max / ps_per_unit),  # tau_g bounds
                (-np.inf, np.inf)                         # time offset
            ]
            fixed_params = [1]  # Index of fixed decay rate parameter
        else:
            bounds = [
                (0, np.inf),                              # amplitude > 0
                (ps_per_unit / self.max_tau, ps_per_unit / self.min_tau),  # decay rate bounds
                (self.tau_g_min / ps_per_unit, self.tau_g_max / ps_per_unit),  # tau_g bounds
                (-np.inf, np.inf)                         # time offset
            ]
            fixed_params = []
        
        return self._levenberg_marquardt_fit(x, y, weights, self.exp_gauss_single, 
                                           beta0, bounds, ps_per_unit, pulse_interval, fixed_params)
    
    def fit_double_exponential(self, x: np.ndarray, y: np.ndarray,
                              ps_per_unit: float, sync_rate: float,
                              beta0: Optional[np.ndarray] = None,
                              weights: Optional[np.ndarray] = None,
                              fix_tau1: Optional[float] = None,
                              fix_tau2: Optional[float] = None) -> Dict[str, Any]:
        """
        Fit double exponential decay model.
        
        Parameters:
        x: time points
        y: photon counts  
        ps_per_unit: picoseconds per time unit
        sync_rate: laser sync rate (Hz)
        beta0: initial parameter guess [amp1, rate1, amp2, rate2, tau_g, t0]
        weights: fitting weights
        fix_tau1: if provided, fix tau1 to this value (in nanoseconds)
        fix_tau2: if provided, fix tau2 to this value (in nanoseconds)
        
        Returns:
        Dictionary with fitting results
        """
        if weights is None:
            weights = np.where(y > 1, 1.0 / np.sqrt(y), 1.0)  # Original method - back to this
            
        # Poisson weights using expected counts (commented out - didn't improve much)
        # if weights is None:
        #     weights = None  # Will be calculated using fitted model
            
        # Calculate pulse interval
        pulse_interval = 1e12 / sync_rate / ps_per_unit
        
        # Initialize parameters if not provided
        if beta0 is None:
            max_val = np.max(y)
            max_idx = np.argmax(y) 
            
            # Better estimate of mean lifetime from data
            weighted_time = np.sum(x * y) / np.sum(y) if np.sum(y) > 0 else len(x) / 2
            est_lifetime_bins = weighted_time - max_idx
            est_lifetime_bins = max(est_lifetime_bins, 10)  # Minimum reasonable lifetime
            
            tau_g = 100 / ps_per_unit
            
            # Set decay rates based on fixed values or estimates
            if fix_tau1 is not None:
                rate1 = ps_per_unit / (fix_tau1 * 1000)
            else:
                rate1 = 1.5 / est_lifetime_bins
                
            if fix_tau2 is not None:
                rate2 = ps_per_unit / (fix_tau2 * 1000)
            else:
                rate2 = 0.5 / est_lifetime_bins
            
            beta0 = np.array([
                max_val * 0.6,                               # amplitude 1 (fast component)
                rate1,                                       # decay rate 1
                max_val * 0.4,                               # amplitude 2 (slow component)
                rate2,                                       # decay rate 2
                tau_g,                                       # instrument response width
                max_idx - tau_g                              # time offset
            ])
        
        # Parameter bounds and fixed parameters
        fixed_params = []
        bounds = [
            (0, np.inf),                              # amplitude 1 > 0
            (ps_per_unit / self.max_tau, ps_per_unit / self.min_tau),  # decay rate 1
            (0, np.inf),                              # amplitude 2 > 0  
            (ps_per_unit / self.max_tau, ps_per_unit / self.min_tau),  # decay rate 2
            (self.tau_g_min / ps_per_unit, self.tau_g_max / ps_per_unit),  # tau_g
            (-np.inf, 10000 / ps_per_unit)            # time offset
        ]
        
        # Fix parameters if tau values are specified
        if fix_tau1 is not None:
            fixed_rate1 = ps_per_unit / (fix_tau1 * 1000)
            bounds[1] = (fixed_rate1, fixed_rate1)
            fixed_params.append(1)
            
        if fix_tau2 is not None:
            fixed_rate2 = ps_per_unit / (fix_tau2 * 1000)
            bounds[3] = (fixed_rate2, fixed_rate2)
            fixed_params.append(3)
        
        return self._levenberg_marquardt_fit(x, y, weights, self.exp_gauss_double,
                                           beta0, bounds, ps_per_unit, pulse_interval, fixed_params)
    
    def _levenberg_marquardt_fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                                model_func, beta0: np.ndarray, bounds: list,
                                ps_per_unit: float, pulse_interval: float, fixed_params: list) -> Dict[str, Any]:
        """Internal Levenberg-Marquardt fitting implementation."""
        n = len(x)
        p = len(beta0)
        
        beta = beta0.copy()
        lambda_param = 0.01
        eps = np.finfo(float).eps
        sqrt_eps = np.sqrt(eps)
        
        # Calculate initial residuals and SSE
        y_fit = model_func(beta, x, pulse_interval)
        residuals = (y - y_fit) * weights
        sse = np.sum(residuals**2)
        sse_old = sse
        
        # Poisson chi-square attempt (commented out - didn't improve significantly)
        # # Calculate proper Poisson weights using expected counts (fitted model)
        # if weights is None:
        #     # Proper Poisson weighting: 1/sqrt(expected_counts)
        #     weights = np.where(y_fit > 1, 1.0 / np.sqrt(y_fit), 1.0)
        # 
        # # Proper Poisson chi-square: sum((observed - expected)^2 / expected)
        # chi_square = np.sum((y - y_fit)**2 / np.where(y_fit > 0, y_fit, 1.0)) / (n - p)
        
        beta_old = beta.copy()
        
        results = {
            'success': False,
            'beta': beta0,
            'residuals': residuals,
            'fit_curve': y_fit,
            'chi_square': sse / (n - p),  # Original calculation - back to this
            'iterations': 0,
            'message': 'Maximum iterations reached'
        }
        
        for iteration in range(self.max_iter):
            beta_old = beta.copy()
            sse_old = sse
            
            # Calculate Jacobian
            jacobian = self.calculate_jacobian(beta, x, model_func, fixed_params, pulse_interval)
            jacobian *= weights[:, np.newaxis]  # Apply weights
            
            # Calculate step
            step = self.levenberg_marquardt_step(jacobian, residuals, lambda_param)
            
            # Apply parameter bounds
            beta_new = beta + step
            for i, (lower, upper) in enumerate(bounds):
                beta_new[i] = np.clip(beta_new[i], lower, upper)
            step = beta_new - beta
            
            # Calculate new residuals and SSE
            y_fit_new = model_func(beta_new, x, pulse_interval)
            residuals_new = (y - y_fit_new) * weights
            sse_new = np.sum(residuals_new**2)
            
            # Iterative weight update (commented out - back to original method)
            # # Update weights with current fitted model for proper Poisson weighting
            # if weights is None or True:  # Always update weights for iterative improvement
            #     weights = np.where(y_fit > 1, 1.0 / np.sqrt(y_fit), 1.0)
            # 
            # # Update weights with new fitted model
            # weights_new = np.where(y_fit_new > 1, 1.0 / np.sqrt(y_fit_new), 1.0)
            # residuals_new = (y - y_fit_new) * weights_new
            
            if sse_new < sse_old:
                # Good step - accept and reduce lambda
                lambda_param = max(0.1 * lambda_param, eps)
                beta = beta_new
                residuals = residuals_new
                sse = sse_new
                y_fit = y_fit_new
            else:
                # Bad step - reject and increase lambda
                lambda_param *= 10
                if lambda_param > 1e16:
                    break
                continue
            
            # Check convergence criteria
            if np.linalg.norm(step) < self.beta_tol * (sqrt_eps + np.linalg.norm(beta)):
                results['success'] = True
                results['message'] = 'Converged (parameter tolerance)'
                break
                
            if abs(sse_old - sse) <= self.r_tol * sse:
                results['success'] = True  
                results['message'] = 'Converged (residual tolerance)'
                break
        
        # Calculate lifetime from fitted parameters
        if model_func == self.exp_gauss_single:
            tau_m = 1 / beta[1]  # lifetime = 1 / decay_rate
        else:  # double exponential
            # Mean lifetime calculation for double exponential
            b = beta
            tau_m = (b[0] / b[1]**2 + b[2] / b[3]**2) / (b[0] / b[1] + b[2] / b[3])
        
        results.update({
            'beta': beta,
            'residuals': residuals,
            'fit_curve': y_fit,
            'chi_square': sse / (n - p),  # Original calculation - back to this
            'iterations': iteration + 1,
            'lifetime': tau_m * ps_per_unit / 1000,  # Convert to nanoseconds
        })
        
        return results
    
    def calculate_mean_lifetime_from_map(self, lifetime_map: np.ndarray, 
                                       intensity_map: np.ndarray,
                                       threshold: float = 0) -> float:
        """
        Calculate mean lifetime from lifetime map with intensity weighting.
        
        Parameters:
        lifetime_map: 2D array of lifetime values
        intensity_map: 2D array of intensity values
        threshold: minimum intensity threshold
        
        Returns:
        Mean lifetime value
        """
        mask = intensity_map > threshold
        if not np.any(mask):
            return 0.0
        
        weighted_sum = np.sum(lifetime_map[mask] * intensity_map[mask])
        total_intensity = np.sum(intensity_map[mask])
        
        return weighted_sum / total_intensity if total_intensity > 0 else 0.0


def demo_flim_fitting():
    """Demonstration of FLIM lifetime fitting with simulated data."""
    
    # Simulation parameters
    ps_per_unit = 12.5  # picoseconds per time bin
    sync_rate = 80e6    # 80 MHz laser
    n_points = 256
    
    # Time axis
    x = np.arange(n_points)
    
    # Simulate single exponential decay data with more realistic parameters
    true_lifetime_ns = 2.0  # 2 nanoseconds
    true_decay_rate = ps_per_unit / (true_lifetime_ns * 1000)  # Convert to units^-1
    true_params_single = np.array([1000, true_decay_rate, 4, 50])  # [amp, rate, tau_g, t0]
    pulse_interval = 1e12 / sync_rate / ps_per_unit
    
    fitter = FLIMLifetimeFitter()
    
    # Generate clean signal
    y_clean = fitter.exp_gauss_single(true_params_single, x, pulse_interval)
    
    # Add Poisson noise
    np.random.seed(42)
    y_noisy = np.random.poisson(y_clean * 1000) / 1000  # Scale for realistic photon counts
    
    # Fit single exponential
    result_single = fitter.fit_single_exponential(x, y_noisy, ps_per_unit, sync_rate)
    
    print("Single Exponential Fit Results:")
    print(f"Success: {result_single['success']}")
    print(f"Iterations: {result_single['iterations']}")
    print(f"Chi-square: {result_single['chi_square']:.4f}")
    print(f"Fitted lifetime: {result_single['lifetime']:.2f} ns")
    print(f"True lifetime: {true_lifetime_ns:.2f} ns")
    print(f"Fitted parameters: {result_single['beta']}")
    print(f"True parameters: {true_params_single}")
    
    # Simulate double exponential decay data with more realistic parameters
    true_tau1_ns = 1.0  # Fast component: 1 ns
    true_tau2_ns = 4.0  # Slow component: 4 ns
    true_rate1 = ps_per_unit / (true_tau1_ns * 1000)
    true_rate2 = ps_per_unit / (true_tau2_ns * 1000)
    true_params_double = np.array([600, true_rate1, 400, true_rate2, 4, 50])  # [amp1, rate1, amp2, rate2, tau_g, t0]
    y_clean_double = fitter.exp_gauss_double(true_params_double, x, pulse_interval)
    y_noisy_double = np.random.poisson(y_clean_double * 1000) / 1000
    
    # Fit double exponential
    result_double = fitter.fit_double_exponential(x, y_noisy_double, ps_per_unit, sync_rate)
    
    print(f"\nDouble Exponential Fit Results:")
    print(f"Success: {result_double['success']}")
    print(f"Iterations: {result_double['iterations']}")
    print(f"Chi-square: {result_double['chi_square']:.4f}")
    print(f"Fitted lifetime: {result_double['lifetime']:.2f} ns")
    print(f"Fitted parameters: {result_double['beta']}")
    print(f"True parameters: {true_params_double}")
    
    # Calculate component lifetimes and populations for double exponential
    b = result_double['beta']
    tau1 = ps_per_unit / b[1] / 1000  # ns
    tau2 = ps_per_unit / b[3] / 1000  # ns
    pop1 = b[0] / (b[0] + b[2])
    pop2 = b[2] / (b[0] + b[2])
    
    print(f"Component 1: τ1 = {tau1:.2f} ns, population = {pop1:.2f}")
    print(f"Component 2: τ2 = {tau2:.2f} ns, population = {pop2:.2f}")
    
    # Test fixed tau fitting
    print(f"\n=== Fixed Tau Fitting Tests ===")
    
    # Test single exponential with fixed tau
    print(f"\nSingle Exponential with Fixed τ = {true_lifetime_ns} ns:")
    result_single_fixed = fitter.fit_single_exponential(x, y_noisy, ps_per_unit, sync_rate, 
                                                       fix_tau=true_lifetime_ns)
    print(f"Success: {result_single_fixed['success']}")
    print(f"Iterations: {result_single_fixed['iterations']}")
    print(f"Chi-square: {result_single_fixed['chi_square']:.4f}")
    print(f"Fitted lifetime: {result_single_fixed['lifetime']:.2f} ns (should be {true_lifetime_ns:.2f} ns)")
    
    # Test double exponential with fixed tau1
    print(f"\nDouble Exponential with Fixed τ1 = {true_tau1_ns} ns:")
    result_double_fix1 = fitter.fit_double_exponential(x, y_noisy_double, ps_per_unit, sync_rate,
                                                      fix_tau1=true_tau1_ns)
    print(f"Success: {result_double_fix1['success']}")
    print(f"Iterations: {result_double_fix1['iterations']}")
    print(f"Chi-square: {result_double_fix1['chi_square']:.4f}")
    b_fix1 = result_double_fix1['beta']
    tau1_fix1 = ps_per_unit / b_fix1[1] / 1000
    tau2_fix1 = ps_per_unit / b_fix1[3] / 1000
    print(f"Component 1: τ1 = {tau1_fix1:.2f} ns (fixed), τ2 = {tau2_fix1:.2f} ns")
    
    # Test double exponential with both tau1 and tau2 fixed
    print(f"\nDouble Exponential with Fixed τ1 = {true_tau1_ns} ns and τ2 = {true_tau2_ns} ns:")
    result_double_fix_both = fitter.fit_double_exponential(x, y_noisy_double, ps_per_unit, sync_rate,
                                                          fix_tau1=true_tau1_ns, fix_tau2=true_tau2_ns)
    print(f"Success: {result_double_fix_both['success']}")
    print(f"Iterations: {result_double_fix_both['iterations']}")
    print(f"Chi-square: {result_double_fix_both['chi_square']:.4f}")
    b_fix_both = result_double_fix_both['beta']
    tau1_fix_both = ps_per_unit / b_fix_both[1] / 1000
    tau2_fix_both = ps_per_unit / b_fix_both[3] / 1000
    pop1_fix_both = b_fix_both[0] / (b_fix_both[0] + b_fix_both[2])
    pop2_fix_both = b_fix_both[2] / (b_fix_both[0] + b_fix_both[2])
    print(f"Component 1: τ1 = {tau1_fix_both:.2f} ns (fixed), population = {pop1_fix_both:.2f}")
    print(f"Component 2: τ2 = {tau2_fix_both:.2f} ns (fixed), population = {pop2_fix_both:.2f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Single exponential
    plt.subplot(1, 3, 1)
    plt.semilogy(x, y_noisy, 'b.', alpha=0.6, label='Data')
    plt.semilogy(x, y_clean, 'g-', label='True model')
    plt.semilogy(x, result_single['fit_curve'], 'r-', label='Fitted model')
    plt.xlabel('Time bins')
    plt.ylabel('Photon counts')
    plt.title('Single Exponential Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Double exponential
    plt.subplot(1, 3, 2)
    plt.semilogy(x, y_noisy_double, 'b.', alpha=0.6, label='Data')
    plt.semilogy(x, y_clean_double, 'g-', label='True model')
    plt.semilogy(x, result_double['fit_curve'], 'r-', label='Fitted model')
    plt.xlabel('Time bins')
    plt.ylabel('Photon counts')
    plt.title('Double Exponential Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals
    plt.subplot(1, 3, 3)
    plt.plot(x, result_single['residuals'], 'b-', alpha=0.7, label='Single exp residuals')
    plt.plot(x, result_double['residuals'], 'r-', alpha=0.7, label='Double exp residuals')
    plt.xlabel('Time bins')
    plt.ylabel('Weighted residuals')
    plt.title('Fitting Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fitter, result_single, result_double, result_single_fixed, result_double_fix1, result_double_fix_both


if __name__ == "__main__2":
    # Run demonstration
    fitter, result_single, result_double, result_single_fixed, result_double_fix1, result_double_fix_both = demo_flim_fitting() 

# %%
if __name__ == "__main__":
    """Demonstration of FLIM lifetime fitting with simulated data."""
    
    # Simulation parameters
    # picoseconds per time bin
    ps_per_unit = 12.5
 
    # sync rate
    sync_rate = 80e6
 
    # number of points
    n_points = 256
    
    # Time axis
    x = np.arange(n_points)
    
    # Simulate single exponential decay data with more realistic parameters
    true_lifetime_ns = 2.0  # 2 nanoseconds
    true_decay_rate = ps_per_unit / (true_lifetime_ns * 1000)  # Convert to units^-1
    true_params_single = np.array([1000, true_decay_rate, 4, 50])  # [amp, rate, tau_g, t0]
    pulse_interval = 1e12 / sync_rate / ps_per_unit
    
    fitter = FLIMLifetimeFitter()
    
    # Generate clean signal
    y_clean = fitter.exp_gauss_single(true_params_single, x, pulse_interval)
    
    # Add Poisson noise
    np.random.seed(42)
    y_noisy = np.random.poisson(y_clean * 1000) / 1000  # Scale for realistic photon counts
    
    # Fit single exponential
    result_single = fitter.fit_single_exponential(x, y_noisy, ps_per_unit, sync_rate)
    
    print("Single Exponential Fit Results:")
    print(f"Success: {result_single['success']}")
    print(f"Iterations: {result_single['iterations']}")
    print(f"Chi-square: {result_single['chi_square']:.4f}")
    print(f"Fitted lifetime: {result_single['lifetime']:.2f} ns")
    print(f"True lifetime: {true_lifetime_ns:.2f} ns")
    print(f"Fitted parameters: {result_single['beta']}")
    print(f"True parameters: {true_params_single}")
    
    # Simulate double exponential decay data with more realistic parameters
    true_tau1_ns = 1.0  # Fast component: 1 ns
    true_tau2_ns = 4.0  # Slow component: 4 ns
    true_rate1 = ps_per_unit / (true_tau1_ns * 1000)
    true_rate2 = ps_per_unit / (true_tau2_ns * 1000)
    true_params_double = np.array([600, true_rate1, 400, true_rate2, 4, 50])  # [amp1, rate1, amp2, rate2, tau_g, t0]
    y_clean_double = fitter.exp_gauss_double(true_params_double, x, pulse_interval)
    y_noisy_double = np.random.poisson(y_clean_double * 1000) / 1000
    
    # Fit double exponential
    result_double = fitter.fit_double_exponential(x, y_noisy_double, ps_per_unit, sync_rate)
    
    print(f"\nDouble Exponential Fit Results:")
    print(f"Success: {result_double['success']}")
    print(f"Iterations: {result_double['iterations']}")
    print(f"Chi-square: {result_double['chi_square']:.4f}")
    print(f"Fitted lifetime: {result_double['lifetime']:.2f} ns")
    print(f"Fitted parameters: {result_double['beta']}")
    print(f"True parameters: {true_params_double}")
    
    # Calculate component lifetimes and populations for double exponential
    b = result_double['beta']
    tau1 = ps_per_unit / b[1] / 1000  # ns
    tau2 = ps_per_unit / b[3] / 1000  # ns
    pop1 = b[0] / (b[0] + b[2])
    pop2 = b[2] / (b[0] + b[2])
    
    print(f"Component 1: τ1 = {tau1:.2f} ns, population = {pop1:.2f}")
    print(f"Component 2: τ2 = {tau2:.2f} ns, population = {pop2:.2f}")
    
    # Test fixed tau fitting
    print(f"\n=== Fixed Tau Fitting Tests ===")
    
    # Test single exponential with fixed tau
    print(f"\nSingle Exponential with Fixed τ = {true_lifetime_ns} ns:")
    result_single_fixed = fitter.fit_single_exponential(x, y_noisy, ps_per_unit, sync_rate, 
                                                       fix_tau=true_lifetime_ns)
    print(f"Success: {result_single_fixed['success']}")
    print(f"Iterations: {result_single_fixed['iterations']}")
    print(f"Chi-square: {result_single_fixed['chi_square']:.4f}")
    print(f"Fitted lifetime: {result_single_fixed['lifetime']:.2f} ns (should be {true_lifetime_ns:.2f} ns)")
    
    # Test double exponential with fixed tau1
    print(f"\nDouble Exponential with Fixed τ1 = {true_tau1_ns} ns:")
    result_double_fix1 = fitter.fit_double_exponential(x, y_noisy_double, ps_per_unit, sync_rate,
                                                      fix_tau1=true_tau1_ns)
    print(f"Success: {result_double_fix1['success']}")
    print(f"Iterations: {result_double_fix1['iterations']}")
    print(f"Chi-square: {result_double_fix1['chi_square']:.4f}")
    b_fix1 = result_double_fix1['beta']
    tau1_fix1 = ps_per_unit / b_fix1[1] / 1000
    tau2_fix1 = ps_per_unit / b_fix1[3] / 1000
    print(f"Component 1: τ1 = {tau1_fix1:.2f} ns (fixed), τ2 = {tau2_fix1:.2f} ns")
    
    # Test double exponential with both tau1 and tau2 fixed
    print(f"\nDouble Exponential with Fixed τ1 = {true_tau1_ns} ns and τ2 = {true_tau2_ns} ns:")
    result_double_fix_both = fitter.fit_double_exponential(x, y_noisy_double, ps_per_unit, sync_rate,
                                                          fix_tau1=true_tau1_ns, fix_tau2=true_tau2_ns)
    print(f"Success: {result_double_fix_both['success']}")
    print(f"Iterations: {result_double_fix_both['iterations']}")
    print(f"Chi-square: {result_double_fix_both['chi_square']:.4f}")
    b_fix_both = result_double_fix_both['beta']
    tau1_fix_both = ps_per_unit / b_fix_both[1] / 1000
    tau2_fix_both = ps_per_unit / b_fix_both[3] / 1000
    pop1_fix_both = b_fix_both[0] / (b_fix_both[0] + b_fix_both[2])
    pop2_fix_both = b_fix_both[2] / (b_fix_both[0] + b_fix_both[2])
    print(f"Component 1: τ1 = {tau1_fix_both:.2f} ns (fixed), population = {pop1_fix_both:.2f}")
    print(f"Component 2: τ2 = {tau2_fix_both:.2f} ns (fixed), population = {pop2_fix_both:.2f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Single exponential
    plt.subplot(1, 3, 1)
    plt.semilogy(x, y_noisy, 'b.', alpha=0.6, label='Data')
    plt.semilogy(x, y_clean, 'g-', label='True model')
    plt.semilogy(x, result_single['fit_curve'], 'r-', label='Fitted model')
    plt.xlabel('Time bins')
    plt.ylabel('Photon counts')
    plt.title('Single Exponential Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Double exponential
    plt.subplot(1, 3, 2)
    plt.semilogy(x, y_noisy_double, 'b.', alpha=0.6, label='Data')
    plt.semilogy(x, y_clean_double, 'g-', label='True model')
    plt.semilogy(x, result_double['fit_curve'], 'r-', label='Fitted model')
    plt.xlabel('Time bins')
    plt.ylabel('Photon counts')
    plt.title('Double Exponential Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals
    plt.subplot(1, 3, 3)
    plt.plot(x, result_single['residuals'], 'b-', alpha=0.7, label='Single exp residuals')
    plt.plot(x, result_double['residuals'], 'r-', alpha=0.7, label='Double exp residuals')
    plt.xlabel('Time bins')
    plt.ylabel('Weighted residuals')
    plt.title('Fitting Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print("Done")

# %%
