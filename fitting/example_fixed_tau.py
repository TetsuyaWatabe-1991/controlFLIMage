#%%
#!/usr/bin/env python3
"""
Simple example demonstrating how to use fixed tau (lifetime) fitting with FLIM data.

Usage examples:
1. Fix tau in single exponential fitting
2. Fix tau1 in double exponential fitting  
3. Fix both tau1 and tau2 in double exponential fitting
"""

import numpy as np
import matplotlib.pyplot as plt
from flim_lifetime_fitting import FLIMLifetimeFitter
import os

def example_fixed_tau_fitting():
    """Demonstrate fixed tau fitting capabilities."""
    
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
    result_normal = fitter.fit_single_exponential(x, y_noisy, ps_per_unit, sync_rate)
    print(f"   Fitted lifetime: {result_normal['lifetime']:.2f} ns")
    print(f"   Iterations: {result_normal['iterations']}")
    print(f"   Chi-square: {result_normal['chi_square']:.4f}")
    
    # Example 2: Fix tau to exact value
    print("\n2. Single Exponential with Fixed τ = 2.5 ns:")
    result_fixed = fitter.fit_single_exponential(x, y_noisy, ps_per_unit, sync_rate,
                                                fix_tau=2.5)
    print(f"   Fitted lifetime: {result_fixed['lifetime']:.2f} ns (fixed value)")
    print(f"   Iterations: {result_fixed['iterations']}")
    print(f"   Chi-square: {result_fixed['chi_square']:.4f}")
    
    # Example 3: Fix tau to different value (test constraint)
    print("\n3. Single Exponential with Fixed τ = 3.0 ns (incorrect value):")
    result_wrong = fitter.fit_single_exponential(x, y_noisy, ps_per_unit, sync_rate,
                                                fix_tau=3.0)
    print(f"   Fitted lifetime: {result_wrong['lifetime']:.2f} ns (forced fixed)")
    print(f"   Iterations: {result_wrong['iterations']}")
    print(f"   Chi-square: {result_wrong['chi_square']:.4f} (worse fit expected)")
    
    # Generate double exponential data
    print("\n" + "="*50)
    print("Double Exponential Examples:")
    
    true_params_double = np.array([600, ps_per_unit/(1.5*1000), 400, ps_per_unit/(4.0*1000), 4, 50])
    y_clean_double = fitter.exp_gauss_double(true_params_double, x, pulse_interval)
    y_noisy_double = np.random.poisson(y_clean_double * 800) / 800
    
    # Example 4: Normal double exponential fitting
    print("\n4. Normal Double Exponential Fitting:")
    result_double_normal = fitter.fit_double_exponential(x, y_noisy_double, ps_per_unit, sync_rate)
    if result_double_normal['success']:
        b = result_double_normal['beta']
        tau1 = ps_per_unit / b[1] / 1000
        tau2 = ps_per_unit / b[3] / 1000
        pop1 = b[0] / (b[0] + b[2])
        print(f"   τ1 = {tau1:.2f} ns, τ2 = {tau2:.2f} ns")
        print(f"   Population 1: {pop1:.2f}, Population 2: {1-pop1:.2f}")
    
    # Example 5: Fix tau1 only
    print("\n5. Double Exponential with Fixed τ1 = 1.5 ns:")
    result_fix_tau1 = fitter.fit_double_exponential(x, y_noisy_double, ps_per_unit, sync_rate,
                                                   fix_tau1=1.5)
    if result_fix_tau1['success']:
        b = result_fix_tau1['beta']
        tau1 = ps_per_unit / b[1] / 1000
        tau2 = ps_per_unit / b[3] / 1000
        pop1 = b[0] / (b[0] + b[2])
        print(f"   τ1 = {tau1:.2f} ns (fixed), τ2 = {tau2:.2f} ns (fitted)")
        print(f"   Population 1: {pop1:.2f}, Population 2: {1-pop1:.2f}")
    
    # Example 6: Fix both tau1 and tau2
    print("\n6. Double Exponential with Fixed τ1 = 1.5 ns and τ2 = 4.0 ns:")
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
    
    # Create comprehensive plots
    plot_fitting_results(x, y_noisy, y_clean, result_normal, result_fixed, result_wrong,
                         y_noisy_double, y_clean_double, result_double_normal, 
                         result_fix_tau1, result_fix_both, ps_per_unit, show_true_model=True)
    
    print("\n=== Summary ===")
    print("How to use fixed tau functionality:")
    print("• fix_tau=X.X : Fix tau in single exponential")
    print("• fix_tau1=X.X : Fix tau1 in double exponential") 
    print("• fix_tau2=X.X : Fix tau2 in double exponential")
    print("• Both can be specified simultaneously")
    print("• Units are in nanoseconds (ns)")

def plot_fitting_results(x, y_single, y_clean_single, result_normal, result_fixed, result_wrong,
                        y_double, y_clean_double, result_double_normal, result_fix_tau1, result_fix_both,
                        ps_per_unit, show_true_model=True):
    """Plot comprehensive fitting results comparing different methods."""
    
    # Convert time bins to nanoseconds for x-axis
    time_ns = x * ps_per_unit / 1000
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('FLIM Lifetime Fitting with Fixed Tau Constraints', fontsize=16, fontweight='bold')
    
    # Single exponential plots
    # Plot 1: Normal vs Fixed (correct value)
    ax1 = axes[0, 0]
    ax1.semilogy(time_ns, y_single, 'k.', alpha=0.6, markersize=3, label='Raw data')
    if show_true_model and y_clean_single is not None:
        ax1.semilogy(time_ns, y_clean_single, 'g-', linewidth=2, label='True model')
    if result_normal['success']:
        ax1.semilogy(time_ns, result_normal['fit_curve'], 'b-', linewidth=2, 
                    label=f'Free fit (τ={result_normal["lifetime"]:.2f} ns)')
    if result_fixed['success']:
        ax1.semilogy(time_ns, result_fixed['fit_curve'], 'r--', linewidth=2, 
                    label=f'Fixed fit')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Photon counts')
    ax1.set_title('Single Exp: Free vs Fixed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fixed correct vs Fixed wrong
    ax2 = axes[0, 1]
    ax2.semilogy(time_ns, y_single, 'k.', alpha=0.6, markersize=3, label='Raw data')
    if show_true_model and y_clean_single is not None:
        ax2.semilogy(time_ns, y_clean_single, 'g-', linewidth=2, label='True model')
    if result_fixed['success']:
        ax2.semilogy(time_ns, result_fixed['fit_curve'], 'r-', linewidth=2, 
                    label=f'Fixed fit (χ²={result_fixed["chi_square"]:.3f})')
    if result_wrong['success']:
        ax2.semilogy(time_ns, result_wrong['fit_curve'], 'm--', linewidth=2, 
                    label=f'Wrong fixed (χ²={result_wrong["chi_square"]:.3f})')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Photon counts')
    ax2.set_title('Single Exp: Comparison of Fixed Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Single exponential residuals
    ax3 = axes[0, 2]
    if result_normal['success']:
        ax3.plot(time_ns, result_normal['residuals'], 'b-', alpha=0.7, 
                label=f'Free fit residuals')
    if result_fixed['success']:
        ax3.plot(time_ns, result_fixed['residuals'], 'r-', alpha=0.7, 
                label=f'Fixed fit residuals')
    if result_wrong['success']:
        ax3.plot(time_ns, result_wrong['residuals'], 'm-', alpha=0.7, 
                label=f'Wrong fixed residuals')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Weighted residuals')
    ax3.set_title('Single Exp: Fitting Residuals')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Double exponential plots
    # Plot 4: Normal double exponential
    ax4 = axes[1, 0]
    ax4.semilogy(time_ns, y_double, 'k.', alpha=0.6, markersize=3, label='Raw data')
    if show_true_model and y_clean_double is not None:
        ax4.semilogy(time_ns, y_clean_double, 'g-', linewidth=2, label='True model')
    if result_double_normal['success']:
        ax4.semilogy(time_ns, result_double_normal['fit_curve'], 'b-', linewidth=2, 
                    label=f'Free fit (τ_m={result_double_normal["lifetime"]:.2f} ns)')
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel('Photon counts')
    ax4.set_title('Double Exp: Free Fit')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Fixed tau1 vs Fixed both
    ax5 = axes[1, 1]
    ax5.semilogy(time_ns, y_double, 'k.', alpha=0.6, markersize=3, label='Raw data')
    if show_true_model and y_clean_double is not None:
        ax5.semilogy(time_ns, y_clean_double, 'g-', linewidth=2, label='True model')
    if result_fix_tau1['success']:
        ax5.semilogy(time_ns, result_fix_tau1['fit_curve'], 'r-', linewidth=2, 
                    label=f'Fixed τ1 only')
    if result_fix_both['success']:
        ax5.semilogy(time_ns, result_fix_both['fit_curve'], 'm--', linewidth=2, 
                    label=f'Fixed both τ1,τ2')
    ax5.set_xlabel('Time (ns)')
    ax5.set_ylabel('Photon counts')
    ax5.set_title('Double Exp: Fixed τ1 vs Fixed Both')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Double exponential residuals
    ax6 = axes[1, 2]
    if result_double_normal['success']:
        ax6.plot(time_ns, result_double_normal['residuals'], 'b-', alpha=0.7, 
                label=f'Free fit residuals')
    if result_fix_tau1['success']:
        ax6.plot(time_ns, result_fix_tau1['residuals'], 'r-', alpha=0.7, 
                label=f'Fixed τ1 residuals')
    if result_fix_both['success']:
        ax6.plot(time_ns, result_fix_both['residuals'], 'm-', alpha=0.7, 
                label=f'Fixed both residuals')
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Time (ns)')
    ax6.set_ylabel('Weighted residuals')
    ax6.set_title('Double Exp: Fitting Residuals')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create a summary plot showing parameter comparison
    plot_parameter_comparison(result_normal, result_fixed, result_wrong, 
                            result_double_normal, result_fix_tau1, result_fix_both)

def plot_parameter_comparison(result_normal, result_fixed, result_wrong, 
                            result_double_normal, result_fix_tau1, result_fix_both):
    """Create a bar plot comparing fitted parameters and chi-square values."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Parameter and Chi-Square Comparison', fontsize=14, fontweight='bold')
    
    # Lifetime comparison
    methods = ['Single\n(Free)', 'Single\n(Fixed 2.5)', 'Single\n(Fixed 3.0)', 
               'Double\n(Free)', 'Double\n(Fix τ1)', 'Double\n(Fix Both)']
    
    lifetimes = []
    chi_squares = []
    colors = ['blue', 'red', 'magenta', 'cyan', 'orange', 'purple']
    
    # Collect data
    for result in [result_normal, result_fixed, result_wrong, 
                   result_double_normal, result_fix_tau1, result_fix_both]:
        if result and result['success']:
            lifetimes.append(result['lifetime'])
            chi_squares.append(result['chi_square'])
        else:
            lifetimes.append(0)
            chi_squares.append(float('inf'))
    
    # Plot lifetimes
    bars1 = ax1.bar(range(len(methods)), lifetimes, color=colors, alpha=0.7)
    ax1.set_xlabel('Fitting Method')
    ax1.set_ylabel('Mean Lifetime (ns)')
    ax1.set_title('Fitted Lifetimes')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, lifetime in zip(bars1, lifetimes):
        if lifetime > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{lifetime:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot chi-squares
    chi_squares_plot = [chi if chi != float('inf') else max([c for c in chi_squares if c != float('inf')]) * 2 
                       for chi in chi_squares]
    bars2 = ax2.bar(range(len(methods)), chi_squares_plot, color=colors, alpha=0.7)
    ax2.set_xlabel('Fitting Method')
    ax2.set_ylabel('Chi-Square')
    ax2.set_title('Goodness of Fit (Chi-Square)')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, chi in zip(bars2, chi_squares):
        if chi != float('inf'):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{chi:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__2":
    example_fixed_tau_fitting() 

# %%
import sys
sys.path.append("..")
from controlFLIMage.FLIMageFileReader2 import FileReader

filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240315\EGFP_004.flim"
filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240315\mStayGold_ftractin_004.flim"
filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240315\NowGFP_009.flim"
filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240315\BrUSLEE_002.flim"
filepaths = [
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240315\EGFP_004.flim",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240315\mStayGold_ftractin_004.flim",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240315\NowGFP_009.flim",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240315\BrUSLEE_002.flim",
    ]

ch_1or2 = 2
photon_threshold = 15
lifetime_result_dict = {}
for filepath in filepaths:
    ImageInfo = FileReader()
    ImageInfo.read_imageFile(filepath, True) 
    imagearray=np.array(ImageInfo.image)
    print(imagearray.shape)
    ch = ch_1or2 - 1
    TwoD_image = np.sum(imagearray[:,:,ch,:,:,:], axis = (0,1,4))
    print(TwoD_image.shape)
    mask = TwoD_image <= photon_threshold

    masked_imagearray = imagearray.copy()
    masked_imagearray[0, 0, ch, mask, :] = 0

    vmax = np.max(np.sum(imagearray[:,:,ch,:,:,:], axis = (0,1,4)))/10

    plt.figure(figsize = (10, 4))
    filename = os.path.basename(filepath)
    
    plt.suptitle(f"{filename}  Photon threshold = {photon_threshold}", y=0.95)
    
    plt.subplot(1, 2, 1)
    plt.title("Original image")
    plt.imshow(np.sum(imagearray[:,:,ch,:,:,:], axis = (0,1,4)), 
    cmap = "gray",
                vmin = 0, vmax = vmax)
    plt.subplot(1, 2, 2)
    plt.title("Photon thresholded image")
    plt.imshow(np.sum(masked_imagearray[:,:,ch,:,:,:], axis = (0,1,4)), 
    cmap = "gray",
                vmin = 0, vmax = vmax)
    plt.tight_layout()
    plt.show()

    one_dim_photon_counts = np.sum(masked_imagearray[:,:,ch,:,:,:], axis = (0,1,2,3))

    # Setup parameters
    sync_rate = 80e6    # 80 MHz laser
    expected_lifetime = 2.6

    fix_tau1 = 3.174
    fix_tau2 = 0.531

    ps_per_unit = (10**12)/sync_rate/len(one_dim_photon_counts)


    result_double_normal = fitter.fit_double_exponential(x, one_dim_photon_counts, ps_per_unit, sync_rate)
    free_2component_fit_tau = result_double_normal["lifetime"]
    lifetime_result_dict[filepath] = free_2component_fit_tau
for filepath, lifetime in lifetime_result_dict.items():
    print(f"{filepath}: {lifetime:.2f} ns")
# result_fix_both = fitter.fit_double_exponential(x, one_dim_photon_counts, ps_per_unit, sync_rate,
#                                                    fix_tau1=fixed_tau1, fix_tau2=fixed_tau2)
# fixed_lifetime = result_fix_both["lifetime"]
# beta = result_fix_both["beta"]
# amp1 = beta[0]
# rate1 = beta[1]
# amp2 = beta[2]
# rate2 = beta[3]
# tau_g = beta[4]
# t0 = beta[5]

# print(f"   fixed_lifetime = {fixed_lifetime:.2f} ns")

# %%
