#%%
#!C:\Users\WatabeT\Documents\Git\venv\Scripts\python.exe
"""
Simple example demonstrating how to use fixed tau (lifetime) fitting with FLIM data.

Usage examples:
1. Fix tau in single exponential fitting
2. Fix tau1 in double exponential fitting  
3. Fix both tau1 and tau2 in double exponential fitting
4. Fix tau1, tau2, tau3 in triple exponential fitting
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
import sys

# Force reload the module to ensure we get the latest version
# This is especially important in Jupyter notebooks
modules_to_reload = [key for key in sys.modules.keys() if 'flim_lifetime_fitting' in key]
for module in modules_to_reload:
    del sys.modules[module]

# Also try to reload any cached bytecode
try:
    importlib.invalidate_caches()
except:
    pass

from flim_lifetime_fitting_simple import FLIMLifetimeFitter

# Verify that all required methods exist
import inspect
required_methods = ['fit_single_exponential', 'fit_double_exponential', 'fit_triple_exponential']
missing_methods = []

for method_name in required_methods:
    if not hasattr(FLIMLifetimeFitter, method_name):
        missing_methods.append(method_name)

if missing_methods:
    print(f"ERROR: Missing methods: {missing_methods}")
    print("Please restart your Python kernel/Jupyter notebook.")
    raise ImportError(f"Required methods missing: {missing_methods}")

# Check parameters for each method
sig_single = inspect.signature(FLIMLifetimeFitter.fit_single_exponential)
sig_double = inspect.signature(FLIMLifetimeFitter.fit_double_exponential)
sig_triple = inspect.signature(FLIMLifetimeFitter.fit_triple_exponential)

print(f"fit_single_exponential parameters: {list(sig_single.parameters.keys())}")
print(f"fit_double_exponential parameters: {list(sig_double.parameters.keys())}")
print(f"fit_triple_exponential parameters: {list(sig_triple.parameters.keys())}")

# Verify fix_tau parameters exist
if 'fix_tau' not in sig_single.parameters:
    print("WARNING: fix_tau parameter not found in fit_single_exponential!")
if 'fix_tau1' not in sig_double.parameters:
    print("WARNING: fix_tau1 parameter not found in fit_double_exponential!")
if 'fix_tau1' not in sig_triple.parameters:
    print("WARNING: fix_tau1 parameter not found in fit_triple_exponential!")

print("✅ All required methods and parameters are available!")

def example_fixed_tau_fitting():
    """Demonstrate fixed tau fitting capabilities."""
    
    try:
        # Setup parameters
        ps_per_unit = 12.5  # picoseconds per time bin
        sync_rate = 80e6    # 80 MHz laser
        n_points = 256
        x = np.arange(n_points)
        
        # Create fitter instance
        fitter = FLIMLifetimeFitter()
        
        # Generate some sample data (single exponential with tau = 2.5 ns)
        true_params = np.array([1000, ps_per_unit / (2.5 * 1000), 4, 50])
        pulse_interval = 1e12 / sync_rate / ps_per_unit
        y_clean = fitter.exp_gauss_single(true_params, x, pulse_interval)
        
        # Add noise
        np.random.seed(123)
        y_noisy = np.random.poisson(y_clean * 1000) / 1000
        
        print("=== Fixed Tau Fitting Examples ===\n")
        
        # Example 1: Normal fitting (no constraints)
        print("1. Normal Single Exponential Fitting:")
        try:
            result_normal = fitter.fit_single_exponential(x, y_noisy, ps_per_unit, sync_rate)
            if result_normal['success']:
                print(f"   Fitted lifetime: {result_normal['lifetime']:.2f} ns")
                print(f"   Iterations: {result_normal['iterations']}")
                print(f"   Chi-square: {result_normal['chi_square']:.4f}")
            else:
                print(f"   Fitting failed: {result_normal['message']}")
        except Exception as e:
            print(f"   Error: {e}")
            result_normal = {'success': False, 'lifetime': 0, 'iterations': 0, 'chi_square': float('inf')}
        
        # Example 2: Fix tau to exact value
        print("\n2. Single Exponential with Fixed τ = 2.5 ns:")
        try:
            result_fixed = fitter.fit_single_exponential(x, y_noisy, ps_per_unit, sync_rate,
                                                        fix_tau=2.5)
            if result_fixed['success']:
                print(f"   Fitted lifetime: {result_fixed['lifetime']:.2f} ns (fixed value)")
                print(f"   Iterations: {result_fixed['iterations']}")
                print(f"   Chi-square: {result_fixed['chi_square']:.4f}")
            else:
                print(f"   Fitting failed: {result_fixed['message']}")
        except Exception as e:
            print(f"   Error: {e}")
            result_fixed = {'success': False, 'lifetime': 0, 'iterations': 0, 'chi_square': float('inf')}
        
        # Example 3: Fix tau to different value (test constraint)
        print("\n3. Single Exponential with Fixed τ = 3.0 ns (incorrect value):")
        try:
            result_wrong = fitter.fit_single_exponential(x, y_noisy, ps_per_unit, sync_rate,
                                                        fix_tau=3.0)
            if result_wrong['success']:
                print(f"   Fitted lifetime: {result_wrong['lifetime']:.2f} ns (forced fixed)")
                print(f"   Iterations: {result_wrong['iterations']}")
                print(f"   Chi-square: {result_wrong['chi_square']:.4f} (worse fit expected)")
            else:
                print(f"   Fitting failed: {result_wrong['message']}")
        except Exception as e:
            print(f"   Error: {e}")
            result_wrong = {'success': False, 'lifetime': 0, 'iterations': 0, 'chi_square': float('inf')}
        
        # Generate double exponential data
        print("\n" + "="*50)
        print("Double Exponential Examples:")
        
        true_params_double = np.array([600, ps_per_unit/(1.5*1000), 400, ps_per_unit/(4.0*1000), 4, 50])
        y_clean_double = fitter.exp_gauss_double(true_params_double, x, pulse_interval)
        y_noisy_double = np.random.poisson(y_clean_double * 800) / 800
        
        # Example 4: Normal double exponential fitting
        print("\n4. Normal Double Exponential Fitting:")
        try:
            result_double_normal = fitter.fit_double_exponential(x, y_noisy_double, ps_per_unit, sync_rate)
            if result_double_normal['success']:
                b = result_double_normal['beta']
                tau1 = ps_per_unit / b[1] / 1000
                tau2 = ps_per_unit / b[3] / 1000
                pop1 = b[0] / (b[0] + b[2])
                print(f"   τ1 = {tau1:.2f} ns, τ2 = {tau2:.2f} ns")
                print(f"   Population 1: {pop1:.2f}, Population 2: {1-pop1:.2f}")
            else:
                print(f"   Fitting failed: {result_double_normal['message']}")
        except Exception as e:
            print(f"   Error: {e}")
            result_double_normal = {'success': False, 'beta': [0, 0, 0, 0, 0, 0]}
        
        # Example 5: Fix tau1 only
        print("\n5. Double Exponential with Fixed τ1 = 1.5 ns:")
        try:
            result_fix_tau1 = fitter.fit_double_exponential(x, y_noisy_double, ps_per_unit, sync_rate,
                                                           fix_tau1=1.5)
            if result_fix_tau1['success']:
                b = result_fix_tau1['beta']
                tau1 = ps_per_unit / b[1] / 1000
                tau2 = ps_per_unit / b[3] / 1000
                pop1 = b[0] / (b[0] + b[2])
                print(f"   τ1 = {tau1:.2f} ns (fixed), τ2 = {tau2:.2f} ns (fitted)")
                print(f"   Population 1: {pop1:.2f}, Population 2: {1-pop1:.2f}")
            else:
                print(f"   Fitting failed: {result_fix_tau1['message']}")
        except Exception as e:
            print(f"   Error: {e}")
            result_fix_tau1 = {'success': False, 'beta': [0, 0, 0, 0, 0, 0]}
        
        # Example 6: Fix both tau1 and tau2
        print("\n6. Double Exponential with Fixed τ1 = 1.5 ns and τ2 = 4.0 ns:")
        try:
            result_fix_both = fitter.fit_double_exponential(x, y_noisy_double, ps_per_unit, sync_rate,
                                                           fix_tau1=1.5, fix_tau2=4.0)
            if result_fix_both['success']:
                b = result_fix_both['beta']
                tau1 = ps_per_unit / b[1] / 1000
                tau2 = ps_per_unit / b[3] / 1000
                pop1 = b[0] / (b[0] + b[2])
                print(f"   τ1 = {tau1:.2f} ns (fixed), τ2 = {tau2:.2f} ns (fixed)")
                print(f"   Population 1: {pop1:.2f}, Population 2: {1-pop1:.2f}")
                print(f"   Only amplitudes are fitted!")
            else:
                print(f"   Fitting failed: {result_fix_both['message']}")
        except Exception as e:
            print(f"   Error: {e}")
            result_fix_both = {'success': False, 'beta': [0, 0, 0, 0, 0, 0]}
        
        # Generate triple exponential data
        print("\n" + "="*50)
        print("Triple Exponential Examples:")
        
        true_params_triple = np.array([400, ps_per_unit/(1.2*1000), 300, ps_per_unit/(3.0*1000), 300, ps_per_unit/(6.0*1000), 4, 50])
        y_clean_triple = fitter.exp_gauss_triple(true_params_triple, x, pulse_interval)
        y_noisy_triple = np.random.poisson(y_clean_triple * 600) / 600
        
        # Example 7: Normal triple exponential fitting
        print("\n7. Normal Triple Exponential Fitting:")
        try:
            result_triple_normal = fitter.fit_triple_exponential(x, y_noisy_triple, ps_per_unit, sync_rate)
            if result_triple_normal['success']:
                print(f"   τ1 = {result_triple_normal['tau1']:.2f} ns, τ2 = {result_triple_normal['tau2']:.2f} ns, τ3 = {result_triple_normal['tau3']:.2f} ns")
                print(f"   Population 1: {result_triple_normal['population1']:.2f}, Population 2: {result_triple_normal['population2']:.2f}, Population 3: {result_triple_normal['population3']:.2f}")
                print(f"   Mean lifetime: {result_triple_normal['lifetime']:.2f} ns")
            else:
                print(f"   Fitting failed: {result_triple_normal['message']}")
        except Exception as e:
            print(f"   Error: {e}")
            result_triple_normal = {'success': False, 'tau1': 0, 'tau2': 0, 'tau3': 0, 'population1': 0, 'population2': 0, 'population3': 0, 'lifetime': 0}
        
        # Example 8: Fix tau1 only in triple exponential
        print("\n8. Triple Exponential with Fixed τ1 = 1.2 ns:")
        try:
            result_triple_fix_tau1 = fitter.fit_triple_exponential(x, y_noisy_triple, ps_per_unit, sync_rate,
                                                                   fix_tau1=1.2)
            if result_triple_fix_tau1['success']:
                print(f"   τ1 = {result_triple_fix_tau1['tau1']:.2f} ns (fixed), τ2 = {result_triple_fix_tau1['tau2']:.2f} ns, τ3 = {result_triple_fix_tau1['tau3']:.2f} ns")
                print(f"   Population 1: {result_triple_fix_tau1['population1']:.2f}, Population 2: {result_triple_fix_tau1['population2']:.2f}, Population 3: {result_triple_fix_tau1['population3']:.2f}")
            else:
                print(f"   Fitting failed: {result_triple_fix_tau1['message']}")
        except Exception as e:
            print(f"   Error: {e}")
            result_triple_fix_tau1 = {'success': False, 'tau1': 0, 'tau2': 0, 'tau3': 0, 'population1': 0, 'population2': 0, 'population3': 0}
        
        # Example 9: Fix tau1 and tau2 in triple exponential
        print("\n9. Triple Exponential with Fixed τ1 = 1.2 ns and τ2 = 3.0 ns:")
        try:
            result_triple_fix_tau12 = fitter.fit_triple_exponential(x, y_noisy_triple, ps_per_unit, sync_rate,
                                                                   fix_tau1=1.2, fix_tau2=3.0)
            if result_triple_fix_tau12['success']:
                print(f"   τ1 = {result_triple_fix_tau12['tau1']:.2f} ns (fixed), τ2 = {result_triple_fix_tau12['tau2']:.2f} ns (fixed), τ3 = {result_triple_fix_tau12['tau3']:.2f} ns")
                print(f"   Population 1: {result_triple_fix_tau12['population1']:.2f}, Population 2: {result_triple_fix_tau12['population2']:.2f}, Population 3: {result_triple_fix_tau12['population3']:.2f}")
            else:
                print(f"   Fitting failed: {result_triple_fix_tau12['message']}")
        except Exception as e:
            print(f"   Error: {e}")
            result_triple_fix_tau12 = {'success': False, 'tau1': 0, 'tau2': 0, 'tau3': 0, 'population1': 0, 'population2': 0, 'population3': 0}
        
        # Example 10: Fix all three tau values in triple exponential
        print("\n10. Triple Exponential with Fixed τ1 = 1.2 ns, τ2 = 3.0 ns, and τ3 = 6.0 ns:")
        try:
            result_triple_fix_all = fitter.fit_triple_exponential(x, y_noisy_triple, ps_per_unit, sync_rate,
                                                                 fix_tau1=1.2, fix_tau2=3.0, fix_tau3=6.0)
            if result_triple_fix_all['success']:
                print(f"   τ1 = {result_triple_fix_all['tau1']:.2f} ns (fixed), τ2 = {result_triple_fix_all['tau2']:.2f} ns (fixed), τ3 = {result_triple_fix_all['tau3']:.2f} ns (fixed)")
                print(f"   Population 1: {result_triple_fix_all['population1']:.2f}, Population 2: {result_triple_fix_all['population2']:.2f}, Population 3: {result_triple_fix_all['population3']:.2f}")
                print(f"   Only amplitudes are fitted!")
            else:
                print(f"   Fitting failed: {result_triple_fix_all['message']}")
        except Exception as e:
            print(f"   Error: {e}")
            result_triple_fix_all = {'success': False, 'tau1': 0, 'tau2': 0, 'tau3': 0, 'population1': 0, 'population2': 0, 'population3': 0}
        
        print("\n=== Summary ===")
        print("How to use fixed tau functionality:")
        print("• fix_tau=X.X : Fix tau in single exponential")
        print("• fix_tau1=X.X : Fix tau1 in double/triple exponential") 
        print("• fix_tau2=X.X : Fix tau2 in double/triple exponential")
        print("• fix_tau3=X.X : Fix tau3 in triple exponential")
        print("• Multiple tau values can be specified simultaneously")
        print("• Units are in nanoseconds (ns)")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Please check the following:")
        print("1. Make sure flim_lifetime_fitting_simple.py is up to date")
        print("2. Restart your Python kernel/Jupyter notebook")
        print("3. Check that all required methods exist")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    example_fixed_tau_fitting()
