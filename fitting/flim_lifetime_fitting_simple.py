"""
Fluorescence Lifetime Imaging (FLIM) data fitting using exponential-Gaussian convolution models.

This implementation is based on the C# FLIMage software algorithms, supporting both 
single and double exponential decay models convolved with Gaussian instrument response.

This is a simplified version for demonstration purposes with minimal dependencies.
"""

import math


def erfc_approx(x):
    """Approximate complementary error function implementation."""
    # Approximation based on Abramowitz and Stegun
    if x < 0:
        return 2 - erfc_approx(-x)
    
    # Constants for approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    
    t = 1.0 / (1.0 + p * x)
    erf_x = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    
    return 1 - erf_x


class FLIMLifetimeFitter:
    """
    Simplified FLIM lifetime fitter with basic functionality.
    """
    
    def __init__(self):
        self.max_iter = 300  # Further increased for better convergence
        self.beta_tol = 1e-4  # Further relaxed for better convergence
        self.r_tol = 1e-5     # Further relaxed for better convergence
        self.tau_g_max = 500  # picoseconds
        self.tau_g_min = 60   # picoseconds
        self.max_tau = 10000  # picoseconds
        self.min_tau = 100    # picoseconds
        
    def exp_gauss_single(self, beta, x, pulse_interval):
        """
        Single exponential decay convolved with Gaussian instrument response.
        
        Parameters:
        beta: [amplitude, decay_rate, tau_g, t0]
        x: list of time points
        pulse_interval: laser pulse interval
        
        Returns:
        y: list of model values
        """
        amplitude, decay_rate, tau_g, t0 = beta
        y = []
        
        for xi in x:
            # Main pulse contribution
            exp_arg = tau_g**2 * decay_rate**2 / 2 - (xi - t0) * decay_rate
            exp_arg = max(-700, min(700, exp_arg))
            y1 = amplitude * math.exp(exp_arg)
            y2 = erfc_approx((tau_g**2 * decay_rate - (xi - t0)) / (math.sqrt(2) * tau_g))
            y_main = y1 * y2
            
            # Previous pulse contribution
            exp_arg_prev = tau_g**2 * decay_rate**2 / 2 - (xi - t0 + pulse_interval) * decay_rate
            exp_arg_prev = max(-700, min(700, exp_arg_prev))
            y1_prev = amplitude * math.exp(exp_arg_prev)
            y2_prev = erfc_approx((tau_g**2 * decay_rate - (xi - t0 + pulse_interval)) / (math.sqrt(2) * tau_g))
            y_prev = y1_prev * y2_prev
            
            y.append((y_main + y_prev) / 2)
        
        return y
    
    def exp_gauss_double(self, beta, x, pulse_interval):
        """
        Double exponential decay convolved with Gaussian instrument response.
        
        Parameters:
        beta: [amplitude1, decay_rate1, amplitude2, decay_rate2, tau_g, t0]
        x: list of time points
        pulse_interval: laser pulse interval
        
        Returns:
        y: list of model values
        """
        amp1, rate1, amp2, rate2, tau_g, t0 = beta
        y = []
        
        for xi in x:
            # Component 1 - Main pulse
            exp_arg1 = tau_g**2 * rate1**2 / 2 - (xi - t0) * rate1
            # Clamp exponential argument to prevent overflow
            exp_arg1 = max(-700, min(700, exp_arg1))
            y1_1 = amp1 * math.exp(exp_arg1)
            y2_1 = erfc_approx((tau_g**2 * rate1 - (xi - t0)) / (math.sqrt(2) * tau_g))
            y_main_1 = y1_1 * y2_1
            
            # Component 1 - Previous pulse
            exp_arg1_prev = tau_g**2 * rate1**2 / 2 - (xi - t0 + pulse_interval) * rate1
            exp_arg1_prev = max(-700, min(700, exp_arg1_prev))
            y1_prev_1 = amp1 * math.exp(exp_arg1_prev)
            y2_prev_1 = erfc_approx((tau_g**2 * rate1 - (xi - t0 + pulse_interval)) / (math.sqrt(2) * tau_g))
            y_prev_1 = y1_prev_1 * y2_prev_1
            
            # Component 2 - Main pulse
            exp_arg2 = tau_g**2 * rate2**2 / 2 - (xi - t0) * rate2
            exp_arg2 = max(-700, min(700, exp_arg2))
            y1_2 = amp2 * math.exp(exp_arg2)
            y2_2 = erfc_approx((tau_g**2 * rate2 - (xi - t0)) / (math.sqrt(2) * tau_g))
            y_main_2 = y1_2 * y2_2
            
            # Component 2 - Previous pulse
            exp_arg2_prev = tau_g**2 * rate2**2 / 2 - (xi - t0 + pulse_interval) * rate2
            exp_arg2_prev = max(-700, min(700, exp_arg2_prev))
            y1_prev_2 = amp2 * math.exp(exp_arg2_prev)
            y2_prev_2 = erfc_approx((tau_g**2 * rate2 - (xi - t0 + pulse_interval)) / (math.sqrt(2) * tau_g))
            y_prev_2 = y1_prev_2 * y2_prev_2
            
            y.append(((y_main_1 + y_prev_1) + (y_main_2 + y_prev_2)) / 2)
        
        return y
    
    def exp_gauss_triple(self, beta, x, pulse_interval):
        """
        Triple exponential decay convolved with Gaussian instrument response.
        
        Parameters:
        beta: [amplitude1, decay_rate1, amplitude2, decay_rate2, amplitude3, decay_rate3, tau_g, t0]
        x: list of time points
        pulse_interval: laser pulse interval
        
        Returns:
        y: list of model values
        """
        amp1, rate1, amp2, rate2, amp3, rate3, tau_g, t0 = beta
        y = []
        
        for xi in x:
            # Component 1 - Main pulse
            exp_arg1 = tau_g**2 * rate1**2 / 2 - (xi - t0) * rate1
            exp_arg1 = max(-700, min(700, exp_arg1))
            y1_1 = amp1 * math.exp(exp_arg1)
            y2_1 = erfc_approx((tau_g**2 * rate1 - (xi - t0)) / (math.sqrt(2) * tau_g))
            y_main_1 = y1_1 * y2_1
            
            # Component 1 - Previous pulse
            exp_arg1_prev = tau_g**2 * rate1**2 / 2 - (xi - t0 + pulse_interval) * rate1
            exp_arg1_prev = max(-700, min(700, exp_arg1_prev))
            y1_prev_1 = amp1 * math.exp(exp_arg1_prev)
            y2_prev_1 = erfc_approx((tau_g**2 * rate1 - (xi - t0 + pulse_interval)) / (math.sqrt(2) * tau_g))
            y_prev_1 = y1_prev_1 * y2_prev_1
            
            # Component 2 - Main pulse
            exp_arg2 = tau_g**2 * rate2**2 / 2 - (xi - t0) * rate2
            exp_arg2 = max(-700, min(700, exp_arg2))
            y1_2 = amp2 * math.exp(exp_arg2)
            y2_2 = erfc_approx((tau_g**2 * rate2 - (xi - t0)) / (math.sqrt(2) * tau_g))
            y_main_2 = y1_2 * y2_2
            
            # Component 2 - Previous pulse
            exp_arg2_prev = tau_g**2 * rate2**2 / 2 - (xi - t0 + pulse_interval) * rate2
            exp_arg2_prev = max(-700, min(700, exp_arg2_prev))
            y1_prev_2 = amp2 * math.exp(exp_arg2_prev)
            y2_prev_2 = erfc_approx((tau_g**2 * rate2 - (xi - t0 + pulse_interval)) / (math.sqrt(2) * tau_g))
            y_prev_2 = y1_prev_2 * y2_prev_2
            
            # Component 3 - Main pulse
            exp_arg3 = tau_g**2 * rate3**2 / 2 - (xi - t0) * rate3
            exp_arg3 = max(-700, min(700, exp_arg3))
            y1_3 = amp3 * math.exp(exp_arg3)
            y2_3 = erfc_approx((tau_g**2 * rate3 - (xi - t0)) / (math.sqrt(2) * tau_g))
            y_main_3 = y1_3 * y2_3
            
            # Component 3 - Previous pulse
            exp_arg3_prev = tau_g**2 * rate3**2 / 2 - (xi - t0 + pulse_interval) * rate3
            exp_arg3_prev = max(-700, min(700, exp_arg3_prev))
            y1_prev_3 = amp3 * math.exp(exp_arg3_prev)
            y2_prev_3 = erfc_approx((tau_g**2 * rate3 - (xi - t0 + pulse_interval)) / (math.sqrt(2) * tau_g))
            y_prev_3 = y1_prev_3 * y2_prev_3
            
            y.append(((y_main_1 + y_prev_1) + (y_main_2 + y_prev_2) + (y_main_3 + y_prev_3)) / 2)
        
        return y
    
    def calculate_residuals(self, beta, x, y_data, pulse_interval, weights=None, model_type='single'):
        """Calculate weighted residuals."""
        if model_type == 'single':
            y_model = self.exp_gauss_single(beta, x, pulse_interval)
        elif model_type == 'double':
            y_model = self.exp_gauss_double(beta, x, pulse_interval)
        elif model_type == 'triple':
            y_model = self.exp_gauss_triple(beta, x, pulse_interval)
        else:
            raise ValueError("model_type must be 'single', 'double', or 'triple'")
        
        if weights is None:
            weights = [1.0 / math.sqrt(max(yi, 1)) for yi in y_data]
        
        residuals = [(y_data[i] - y_model[i]) * weights[i] for i in range(len(x))]
        sse = sum(r**2 for r in residuals)
        
        return residuals, sse, y_model
    
    def numerical_jacobian(self, beta, x, y_data, pulse_interval, weights=None, model_type='single'):
        """Calculate Jacobian using finite differences."""
        if weights is None:
            weights = [1.0 / math.sqrt(max(yi, 1)) for yi in y_data]
        
        n = len(x)
        p = len(beta)
        delta_step = self.beta_tol * 0.01
        
        jacobian = [[0.0 for _ in range(p)] for _ in range(n)]
        
        if model_type == 'single':
            y_fit = self.exp_gauss_single(beta, x, pulse_interval)
        elif model_type == 'double':
            y_fit = self.exp_gauss_double(beta, x, pulse_interval)
        elif model_type == 'triple':
            y_fit = self.exp_gauss_triple(beta, x, pulse_interval)
        else:
            raise ValueError("model_type must be 'single', 'double', or 'triple'")
        
        for j in range(p):
            beta_new = beta[:]
            delta = delta_step * abs(beta[j]) if beta[j] != 0 else delta_step
            if delta == 0:
                delta = delta_step
                
            beta_new[j] += delta
            
            if model_type == 'single':
                y_plus = self.exp_gauss_single(beta_new, x, pulse_interval)
            elif model_type == 'double':
                y_plus = self.exp_gauss_double(beta_new, x, pulse_interval)
            elif model_type == 'triple':
                y_plus = self.exp_gauss_triple(beta_new, x, pulse_interval)
            
            for i in range(n):
                jacobian[i][j] = (y_plus[i] - y_fit[i]) / delta * weights[i]
        
        return jacobian
    
    def matrix_multiply(self, A, B):
        """Simple matrix multiplication."""
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        
        if cols_A != rows_B:
            raise ValueError("Matrix dimensions incompatible")
        
        result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
        
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        
        return result
    
    def matrix_transpose(self, A):
        """Transpose matrix."""
        return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
    
    def matrix_add_diagonal(self, A, factor):
        """Add factor * I to diagonal of matrix A."""
        result = [row[:] for row in A]  # Copy matrix
        for i in range(len(A)):
            result[i][i] *= (1 + factor)
        return result
    
    def gauss_elimination(self, A, b):
        """Solve Ax = b using Gaussian elimination with partial pivoting."""
        n = len(A)
        # Create augmented matrix
        aug = [A[i][:] + [b[i]] for i in range(n)]
        
        # Forward elimination
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            aug[i], aug[max_row] = aug[max_row], aug[i]
            
            # Make all rows below this one 0 in current column
            for k in range(i + 1, n):
                if aug[i][i] != 0:
                    factor = aug[k][i] / aug[i][i]
                    for j in range(i, n + 1):
                        aug[k][j] -= factor * aug[i][j]
        
        # Back substitution
        x = [0] * n
        for i in range(n - 1, -1, -1):
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            if aug[i][i] != 0:
                x[i] /= aug[i][i]
        
        return x
    
    def fit_single_exponential(self, x, y, ps_per_unit, sync_rate, beta0=None, fix_tau=None):
        """
        Fit single exponential decay model using simplified Levenberg-Marquardt.
        
        Parameters:
        x: list of time points
        y: list of photon counts
        ps_per_unit: picoseconds per time unit
        sync_rate: laser sync rate (Hz)
        beta0: initial parameter guess [amplitude, decay_rate, tau_g, t0]
        fix_tau: fix tau to specified value (ns)
        
        Returns:
        Dictionary with fitting results
        """
        # Calculate pulse interval
        pulse_interval = 1e12 / sync_rate / ps_per_unit
        
        # Initialize parameters if not provided
        if beta0 is None:
            max_val = max(y)
            max_idx = list(y).index(max_val) if hasattr(y, 'index') else int(y.argmax())
            sum_y = sum(y)
            
            # More conservative initial parameters
            tau_g = 50 / ps_per_unit  # Smaller instrument response width
            estimated_lifetime = 2.5  # Assume 2.5 ns typical lifetime
            decay_rate = ps_per_unit / (estimated_lifetime * 1000)
            
            beta0 = [
                max_val * 1.2,                    # amplitude (slightly higher than max)
                decay_rate,                       # decay rate (conservative estimate)
                tau_g,                            # instrument response width
                max_idx - tau_g                   # time offset (more conservative)
            ]
        
        # Apply fixed tau constraint
        if fix_tau is not None:
            beta0[1] = ps_per_unit / (fix_tau * 1000)  # Convert ns to ps, then to rate
        
        # Poisson weights
        weights = [1.0 / math.sqrt(max(yi, 1)) for yi in y]
        
        beta = beta0[:]
        lambda_param = 0.0001  # Even smaller initial lambda for better convergence
        eps = 1e-15
        
        residuals, sse, y_fit = self.calculate_residuals(beta, x, y, pulse_interval, weights, 'single')
        
        results = {
            'success': False,
            'beta': beta0[:],
            'residuals': residuals,
            'fit_curve': y_fit,
            'chi_square': sse / (len(x) - len(beta)),
            'iterations': 0,
            'message': 'Maximum iterations reached'
        }
        
        for iteration in range(self.max_iter):
            sse_old = sse
            beta_old = beta[:]
            
            # Calculate Jacobian
            jacobian = self.numerical_jacobian(beta, x, y, pulse_interval, weights, 'single')
            
            # Calculate J^T * J and J^T * r
            jt = self.matrix_transpose(jacobian)
            jtj = self.matrix_multiply(jt, jacobian)
            jtr = [sum(jt[i][j] * residuals[j] for j in range(len(residuals))) for i in range(len(jt))]
            
            # Add damping term (Levenberg-Marquardt)
            jtj_damped = self.matrix_add_diagonal(jtj, lambda_param)
            
            try:
                # Solve for step
                step = self.gauss_elimination(jtj_damped, jtr)
                
                # Update parameters with bounds checking
                beta_new = [beta[i] + step[i] for i in range(len(beta))]
                
                # Apply bounds
                beta_new[0] = max(0, beta_new[0])  # amplitude > 0
                
                # Apply tau bounds (unless fixed)
                if fix_tau is None:
                    beta_new[1] = max(ps_per_unit / self.max_tau, 
                                    min(ps_per_unit / self.min_tau, beta_new[1]))  # decay rate bounds
                else:
                    # Keep tau fixed
                    beta_new[1] = ps_per_unit / (fix_tau * 1000)
                
                beta_new[2] = max(self.tau_g_min / ps_per_unit, 
                                min(self.tau_g_max / ps_per_unit, beta_new[2]))  # tau_g bounds
                
                # Calculate new residuals
                residuals_new, sse_new, y_fit_new = self.calculate_residuals(beta_new, x, y, pulse_interval, weights, 'single')
                
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
                
                # Check convergence
                step_norm = math.sqrt(sum(s**2 for s in step))
                beta_norm = math.sqrt(sum(b**2 for b in beta))
                
                if step_norm < self.beta_tol * (math.sqrt(eps) + beta_norm):
                    results['success'] = True
                    results['message'] = 'Converged (parameter tolerance)'
                    break
                    
                if abs(sse_old - sse) <= self.r_tol * sse:
                    results['success'] = True
                    results['message'] = 'Converged (residual tolerance)'
                    break
                    
            except:
                # If solving fails, increase lambda and continue
                lambda_param *= 10
                if lambda_param > 1e16:
                    break
                continue
        
        # Calculate lifetime
        tau_m = 1 / beta[1]  # lifetime = 1 / decay_rate
        lifetime_ns = tau_m * ps_per_unit / 1000  # Convert to nanoseconds
        
        results.update({
            'beta': beta,
            'residuals': residuals,
            'fit_curve': y_fit,
            'chi_square': sse / (len(x) - len(beta)),
            'iterations': iteration + 1,
            'lifetime': lifetime_ns,
            'fix_tau': fix_tau
        })
        
        return results
    
    def fit_double_exponential(self, x, y, ps_per_unit, sync_rate, beta0=None, 
                              fix_tau1=None, fix_tau2=None):
        """
        Fit double exponential decay model using simplified Levenberg-Marquardt.
        
        Parameters:
        x: list of time points
        y: list of photon counts
        ps_per_unit: picoseconds per time unit
        sync_rate: laser sync rate (Hz)
        beta0: initial parameter guess [amplitude1, decay_rate1, amplitude2, decay_rate2, tau_g, t0]
        fix_tau1: fix tau1 to specified value (ns)
        fix_tau2: fix tau2 to specified value (ns)
        
        Returns:
        Dictionary with fitting results
        """
        # Calculate pulse interval
        pulse_interval = 1e12 / sync_rate / ps_per_unit
        
        # Initialize parameters if not provided
        if beta0 is None:
            max_val = max(y)
            max_idx = list(y).index(max_val) if hasattr(y, 'index') else int(y.argmax())
            sum_y = sum(y)
            
            # More conservative initial parameters for double exponential
            tau_g = 20 / ps_per_unit  # Very small instrument response width
            tau1_ns = 1.5  # First component lifetime
            tau2_ns = 4.0  # Second component lifetime
            
            # Use more realistic amplitude ratios based on data characteristics
            beta0 = [
                max_val * 0.8,                                      # amplitude1 (dominant)
                ps_per_unit / (tau1_ns * 1000),                    # decay_rate1
                max_val * 0.2,                                      # amplitude2 (smaller)
                ps_per_unit / (tau2_ns * 1000),                    # decay_rate2
                tau_g,                                              # instrument response width
                max_idx - tau_g * 0.3                              # time offset (very conservative)
            ]
        
        # Apply fixed tau constraints
        if fix_tau1 is not None:
            beta0[1] = ps_per_unit / (fix_tau1 * 1000)  # Convert ns to ps, then to rate
        if fix_tau2 is not None:
            beta0[3] = ps_per_unit / (fix_tau2 * 1000)
        
        # Poisson weights
        weights = [1.0 / math.sqrt(max(yi, 1)) for yi in y]
        
        beta = beta0[:]
        lambda_param = 0.0001  # Even smaller initial lambda for better convergence
        eps = 1e-15
        
        residuals, sse, y_fit = self.calculate_residuals(beta, x, y, pulse_interval, weights, 'double')
        
        results = {
            'success': False,
            'beta': beta0[:],
            'residuals': residuals,
            'fit_curve': y_fit,
            'chi_square': sse / (len(x) - len(beta)),
            'iterations': 0,
            'message': 'Maximum iterations reached'
        }
        
        for iteration in range(self.max_iter):
            sse_old = sse
            beta_old = beta[:]
            
            # Calculate Jacobian
            jacobian = self.numerical_jacobian(beta, x, y, pulse_interval, weights, 'double')
            
            # Calculate J^T * J and J^T * r
            jt = self.matrix_transpose(jacobian)
            jtj = self.matrix_multiply(jt, jacobian)
            jtr = [sum(jt[i][j] * residuals[j] for j in range(len(residuals))) for i in range(len(jt))]
            
            # Add damping term (Levenberg-Marquardt)
            jtj_damped = self.matrix_add_diagonal(jtj, lambda_param)
            
            try:
                # Solve for step
                step = self.gauss_elimination(jtj_damped, jtr)
                
                # Update parameters with bounds checking
                beta_new = [beta[i] + step[i] for i in range(len(beta))]
                
                # Apply bounds
                beta_new[0] = max(0, beta_new[0])  # amplitude1 > 0
                beta_new[2] = max(0, beta_new[2])  # amplitude2 > 0
                
                # Apply tau bounds (unless fixed)
                if fix_tau1 is None:
                    beta_new[1] = max(ps_per_unit / self.max_tau, 
                                    min(ps_per_unit / self.min_tau, beta_new[1]))
                if fix_tau2 is None:
                    beta_new[3] = max(ps_per_unit / self.max_tau, 
                                    min(ps_per_unit / self.min_tau, beta_new[3]))
                
                beta_new[4] = max(self.tau_g_min / ps_per_unit, 
                                min(self.tau_g_max / ps_per_unit, beta_new[4]))  # tau_g bounds
                
                # Calculate new residuals
                residuals_new, sse_new, y_fit_new = self.calculate_residuals(beta_new, x, y, pulse_interval, weights, 'double')
                
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
                
                # Check convergence
                step_norm = math.sqrt(sum(s**2 for s in step))
                beta_norm = math.sqrt(sum(b**2 for b in beta))
                
                if step_norm < self.beta_tol * (math.sqrt(eps) + beta_norm):
                    results['success'] = True
                    results['message'] = 'Converged (parameter tolerance)'
                    break
                    
                if abs(sse_old - sse) <= self.r_tol * sse:
                    results['success'] = True
                    results['message'] = 'Converged (residual tolerance)'
                    break
                    
            except:
                # If solving fails, increase lambda and continue
                lambda_param *= 10
                if lambda_param > 1e16:
                    break
                continue
        
        # Calculate lifetimes and populations
        tau1 = ps_per_unit / beta[1] / 1000  # Convert to nanoseconds
        tau2 = ps_per_unit / beta[3] / 1000
        
        # Calculate mean lifetime (amplitude-weighted)
        total_amp = beta[0] + beta[2]
        mean_lifetime = (beta[0] * tau1 + beta[2] * tau2) / total_amp
        
        # Calculate populations
        pop1 = beta[0] / total_amp
        pop2 = beta[2] / total_amp
        
        results.update({
            'beta': beta,
            'residuals': residuals,
            'fit_curve': y_fit,
            'chi_square': sse / (len(x) - len(beta)),
            'iterations': iteration + 1,
            'lifetime': mean_lifetime,
            'tau1': tau1,
            'tau2': tau2,
            'population1': pop1,
            'population2': pop2,
            'fix_tau1': fix_tau1,
            'fix_tau2': fix_tau2
        })
        
        return results
    
    def fit_triple_exponential(self, x, y, ps_per_unit, sync_rate, beta0=None, 
                              fix_tau1=None, fix_tau2=None, fix_tau3=None):
        """
        Fit triple exponential decay model using simplified Levenberg-Marquardt.
        
        Parameters:
        x: list of time points
        y: list of photon counts
        ps_per_unit: picoseconds per time unit
        sync_rate: laser sync rate (Hz)
        beta0: initial parameter guess [amplitude1, decay_rate1, amplitude2, decay_rate2, amplitude3, decay_rate3, tau_g, t0]
        fix_tau1: fix tau1 to specified value (ns)
        fix_tau2: fix tau2 to specified value (ns)
        fix_tau3: fix tau3 to specified value (ns)
        
        Returns:
        Dictionary with fitting results
        """
        # Calculate pulse interval
        pulse_interval = 1e12 / sync_rate / ps_per_unit
        
        # Initialize parameters if not provided
        if beta0 is None:
            max_val = max(y)
            max_idx = list(y).index(max_val) if hasattr(y, 'index') else int(y.argmax())
            sum_y = sum(y)
            
            # More conservative initial parameters for triple exponential
            tau_g = 20 / ps_per_unit  # Very small instrument response width
            tau1_ns = 1.2  # First component lifetime
            tau2_ns = 3.0  # Second component lifetime
            tau3_ns = 6.0  # Third component lifetime
            
            # Use more realistic amplitude ratios based on data characteristics
            beta0 = [
                max_val * 0.6,                                      # amplitude1 (dominant)
                ps_per_unit / (tau1_ns * 1000),                    # decay_rate1
                max_val * 0.3,                                      # amplitude2 (medium)
                ps_per_unit / (tau2_ns * 1000),                    # decay_rate2
                max_val * 0.1,                                      # amplitude3 (smallest)
                ps_per_unit / (tau3_ns * 1000),                    # decay_rate3
                tau_g,                                              # instrument response width
                max_idx - tau_g * 0.3                              # time offset (very conservative)
            ]
        
        # Apply fixed tau constraints
        if fix_tau1 is not None:
            beta0[1] = ps_per_unit / (fix_tau1 * 1000)  # Convert ns to ps, then to rate
        if fix_tau2 is not None:
            beta0[3] = ps_per_unit / (fix_tau2 * 1000)
        if fix_tau3 is not None:
            beta0[5] = ps_per_unit / (fix_tau3 * 1000)
        
        # Poisson weights
        weights = [1.0 / math.sqrt(max(yi, 1)) for yi in y]
        
        beta = beta0[:]
        lambda_param = 0.0001  # Even smaller initial lambda for better convergence
        eps = 1e-15
        
        residuals, sse, y_fit = self.calculate_residuals(beta, x, y, pulse_interval, weights, 'triple')
        
        results = {
            'success': False,
            'beta': beta0[:],
            'residuals': residuals,
            'fit_curve': y_fit,
            'chi_square': sse / (len(x) - len(beta)),
            'iterations': 0,
            'message': 'Maximum iterations reached'
        }
        
        for iteration in range(self.max_iter):
            sse_old = sse
            beta_old = beta[:]
            
            # Calculate Jacobian
            jacobian = self.numerical_jacobian(beta, x, y, pulse_interval, weights, 'triple')
            
            # Calculate J^T * J and J^T * r
            jt = self.matrix_transpose(jacobian)
            jtj = self.matrix_multiply(jt, jacobian)
            jtr = [sum(jt[i][j] * residuals[j] for j in range(len(residuals))) for i in range(len(jt))]
            
            # Add damping term (Levenberg-Marquardt)
            jtj_damped = self.matrix_add_diagonal(jtj, lambda_param)
            
            try:
                # Solve for step
                step = self.gauss_elimination(jtj_damped, jtr)
                
                # Update parameters with bounds checking
                beta_new = [beta[i] + step[i] for i in range(len(beta))]
                
                # Apply bounds
                beta_new[0] = max(0, beta_new[0])  # amplitude1 > 0
                beta_new[2] = max(0, beta_new[2])  # amplitude2 > 0
                beta_new[4] = max(0, beta_new[4])  # amplitude3 > 0
                
                # Apply tau bounds (unless fixed)
                if fix_tau1 is None:
                    beta_new[1] = max(ps_per_unit / self.max_tau, 
                                    min(ps_per_unit / self.min_tau, beta_new[1]))
                if fix_tau2 is None:
                    beta_new[3] = max(ps_per_unit / self.max_tau, 
                                    min(ps_per_unit / self.min_tau, beta_new[3]))
                if fix_tau3 is None:
                    beta_new[5] = max(ps_per_unit / self.max_tau, 
                                    min(ps_per_unit / self.min_tau, beta_new[5]))
                
                beta_new[6] = max(self.tau_g_min / ps_per_unit, 
                                min(self.tau_g_max / ps_per_unit, beta_new[6]))  # tau_g bounds
                
                # Calculate new residuals
                residuals_new, sse_new, y_fit_new = self.calculate_residuals(beta_new, x, y, pulse_interval, weights, 'triple')
                
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
                
                # Check convergence
                step_norm = math.sqrt(sum(s**2 for s in step))
                beta_norm = math.sqrt(sum(b**2 for b in beta))
                
                if step_norm < self.beta_tol * (math.sqrt(eps) + beta_norm):
                    results['success'] = True
                    results['message'] = 'Converged (parameter tolerance)'
                    break
                    
                if abs(sse_old - sse) <= self.r_tol * sse:
                    results['success'] = True
                    results['message'] = 'Converged (residual tolerance)'
                    break
                    
            except:
                # If solving fails, increase lambda and continue
                lambda_param *= 10
                if lambda_param > 1e16:
                    break
                continue
        
        # Calculate lifetimes and populations
        tau1 = ps_per_unit / beta[1] / 1000  # Convert to nanoseconds
        tau2 = ps_per_unit / beta[3] / 1000
        tau3 = ps_per_unit / beta[5] / 1000
        
        # Calculate mean lifetime (amplitude-weighted)
        total_amp = beta[0] + beta[2] + beta[4]
        mean_lifetime = (beta[0] * tau1 + beta[2] * tau2 + beta[4] * tau3) / total_amp
        
        # Calculate populations
        pop1 = beta[0] / total_amp
        pop2 = beta[2] / total_amp
        pop3 = beta[4] / total_amp
        
        results.update({
            'beta': beta,
            'residuals': residuals,
            'fit_curve': y_fit,
            'chi_square': sse / (len(x) - len(beta)),
            'iterations': iteration + 1,
            'lifetime': mean_lifetime,
            'tau1': tau1,
            'tau2': tau2,
            'tau3': tau3,
            'population1': pop1,
            'population2': pop2,
            'population3': pop3,
            'fix_tau1': fix_tau1,
            'fix_tau2': fix_tau2,
            'fix_tau3': fix_tau3
        })
        
        return results


def demo_simple():
    """Simple demonstration without external dependencies."""
    print("FLIM Lifetime Fitting Demonstration")
    print("=" * 40)
    
    # Simulation parameters
    ps_per_unit = 12.5  # picoseconds per time bin
    sync_rate = 80e6    # 80 MHz laser
    n_points = 128
    
    # Time axis
    x = list(range(n_points))
    
    # True parameters for single exponential
    true_params = [1000, 0.4, 4, 50]  # [amp, rate, tau_g, t0]
    
    fitter = FLIMLifetimeFitter()
    
    # Calculate pulse interval
    pulse_interval = 1e12 / sync_rate / ps_per_unit
    
    # Generate clean signal
    y_clean = fitter.exp_gauss_single(true_params, x, pulse_interval)
    
    # Add some simple noise (not Poisson for simplicity)
    import random
    random.seed(42)
    noise_level = 0.1
    y_noisy = [max(1, y + random.gauss(0, y * noise_level)) for y in y_clean]
    
    print(f"Simulated data with {n_points} time points")
    print(f"True lifetime: {ps_per_unit / true_params[1] / 1000:.2f} ns")
    print(f"True parameters: {true_params}")
    print()
    
    # Fit the data
    print("Fitting single exponential model...")
    result = fitter.fit_single_exponential(x, y_noisy, ps_per_unit, sync_rate)
    
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Chi-square: {result['chi_square']:.4f}")
    print(f"Fitted lifetime: {result['lifetime']:.2f} ns")
    print(f"Fitted parameters: {[f'{p:.3f}' for p in result['beta']]}")
    
    # Calculate error in lifetime estimation
    true_lifetime = ps_per_unit / true_params[1] / 1000
    error_percent = abs(result['lifetime'] - true_lifetime) / true_lifetime * 100
    print(f"Lifetime error: {error_percent:.1f}%")
    
    # Calculate some fit statistics
    max_residual = max(abs(r) for r in result['residuals'])
    mean_residual = sum(abs(r) for r in result['residuals']) / len(result['residuals'])
    print(f"Max residual: {max_residual:.3f}")
    print(f"Mean |residual|: {mean_residual:.3f}")
    
    return fitter, result


if __name__ == "__main__":
    # Run demonstration
    fitter, result = demo_simple() 