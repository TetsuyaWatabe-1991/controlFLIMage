import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import median_filter
from datetime import datetime
from FLIMageFileReader2 import FileReader
from FLIMageAlignment import get_flimfile_list

def calculate_laser_power_settings(laser_mW_ms, power_slope, power_intercept, 
                                 from_Thorlab_to_coherent_factor=1/3, max_power=100):
    """
    Calculate uncaging power settings from desired mW values.
    
    Args:
        laser_mW_ms: List of [mW, ms] pairs
        power_slope: Slope of power calibration
        power_intercept: Intercept of power calibration
        from_Thorlab_to_coherent_factor: Conversion factor
        max_power: Maximum allowed power percentage
    
    Returns:
        List of [power_percentage, duration_ms] pairs
    """
    unc_pow_dur = []
    for each_mw_ms in laser_mW_ms:
        mW_in_Thorlabs = each_mw_ms[0]/from_Thorlab_to_coherent_factor
        pow_from_mw = int((mW_in_Thorlabs - power_intercept)/power_slope)
        if pow_from_mw <= max_power:    
            unc_pow_dur.append([pow_from_mw, each_mw_ms[1]])
        else:
            print(f"pow_from_mw = {pow_from_mw} is over {max_power}")
    return unc_pow_dur

def process_flim_image(each_file, power_slope, power_intercept, 
                      from_Thorlab_to_coherent_factor=1/3):
    """
    Process a single FLIM image file and extract relevant data.
    
    Args:
        each_file: Path to FLIM file
        power_slope: Slope of power calibration
        power_intercept: Intercept of power calibration
        from_Thorlab_to_coherent_factor: Conversion factor
    
    Returns:
        Dictionary containing processed image data
    """
    uncaging_iminfo = FileReader()
    uncaging_iminfo.read_imageFile(each_file, True) 
    
    imagearray = np.array(uncaging_iminfo.image)
    
    # Check if image has expected shape
    allowed_shape_0th_list = [4, 32, 33, 34]
    if (imagearray.shape)[0] not in allowed_shape_0th_list:
        print(f"Skipped {each_file} - shape {imagearray.shape} not in {allowed_shape_0th_list}")
        return None
    
    # Extract image data
    uncaging_x_y_0to1 = uncaging_iminfo.statedict["State.Uncaging.Position"]
    uncaging_pow = uncaging_iminfo.statedict["State.Uncaging.Power"]
    pulseWidth = uncaging_iminfo.statedict["State.Uncaging.pulseWidth"]
    
    center_y = imagearray.shape[-2] * uncaging_x_y_0to1[1]
    center_x = imagearray.shape[-3] * uncaging_x_y_0to1[0]

    if imagearray.shape[0] in [4, 33, 34]:
        GCpre = imagearray[0,0,0,:,:,:].sum(axis=-1)
        GCunc = imagearray[3,0,0,:,:,:].sum(axis=-1)
        Tdpre = imagearray[0,0,1,:,:,:].sum(axis=-1)
    elif imagearray.shape[0] in [32]:
        GCpre = imagearray[8*0 + 1 : 8*1, 0,0,:,:,:].sum(axis=-1).sum(axis=0)
        GCunc = imagearray[8*3 + 1 : 8*4, 0,0,:,:,:].sum(axis=-1).sum(axis=0)
        Tdpre = imagearray[8*0 + 1 : 8*1, 0,1,:,:,:].sum(axis=-1).sum(axis=0)
    assert len(GCpre.shape) == 2 #Image should be 2D

    # Apply median filter
    GC_pre_med = median_filter(GCpre, size=3)
    GC_unc_med = median_filter(GCunc, size=3)
    
    # Calculate F/F0
    GCF_F0 = (GC_unc_med/GC_pre_med)
    GCF_F0[GC_pre_med == 0] = 0

    # Calculate power in mW
    pow_mw = power_slope * uncaging_pow + power_intercept
    pow_mw_coherent = pow_mw * from_Thorlab_to_coherent_factor
    pow_mw_round = round(pow_mw_coherent, 1)
    
    return {
        'imagearray': imagearray,
        'statedict': uncaging_iminfo.statedict,
        'GCF_F0': GCF_F0,
        'Tdpre': Tdpre,
        'center_x': center_x,
        'center_y': center_y,
        'pow_mw_round': pow_mw_round,
        'pulseWidth': pulseWidth
    }

def plot_single_image(data, F_F0_vmin=1, F_F0_vmax=10, save_path=None):
    """
    Plot a single F/F0 image.
    
    Args:
        data: Dictionary containing processed image data
        F_F0_vmin: Minimum value for color scale
        F_F0_vmax: Maximum value for color scale
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(data['GCF_F0'], cmap="inferno", vmin=F_F0_vmin, vmax=F_F0_vmax)
    plt.plot(data['center_x'], data['center_y'], 'ro', markersize=2)   
    plt.title(f"{data['pow_mw_round']} mW, {data['pulseWidth']} ms")  
    plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()

def plot_multiple_images(result_img_dict, F_F0_vmin=1, F_F0_vmax=10, save_path=None):
    """
    Plot multiple F/F0 images in a grid with tdTomato reference.
    
    Args:
        result_img_dict: Dictionary of processed image data
        F_F0_vmin: Minimum value for color scale
        F_F0_vmax: Maximum value for color scale
        save_path: Path to save the plot (optional)
    """
    n_imgs = len(result_img_dict)
    each_fig_size = 2
    fig = plt.figure(figsize=(each_fig_size*(n_imgs+2), each_fig_size))
    gs = gridspec.GridSpec(1, n_imgs+2, width_ratios=[0.9] + [1]*n_imgs + [0.08], wspace=0.05)

    for nth_plot, (each_file, data) in enumerate(result_img_dict.items()):
        # Plot Tdpre at far left, only for the first image
        if nth_plot == 0:
            ax_td = fig.add_subplot(gs[0, 0])
            ax_td.imshow(data['Tdpre'], cmap='gray')
            ax_td.plot(data['center_x'], data['center_y'], 'c+', markersize=5)
            ax_td.set_title('tdTomato')
            ax_td.axis('off')

        # Plot F/F0 images in columns 1,2,3,...
        ax = fig.add_subplot(gs[0, nth_plot+1])
        im = ax.imshow(data['GCF_F0'], cmap="inferno", vmin=F_F0_vmin, vmax=F_F0_vmax)
        ax.plot(data['center_x'], data['center_y'], 'c+', markersize=5)
        ax.set_title(f"{data['pow_mw_round']} mW, {data['pulseWidth']} ms")
        ax.axis('off')

    # Color bar at the far right
    ax_cbar = fig.add_subplot(gs[0, -1])
    norm = Normalize(vmin=F_F0_vmin, vmax=F_F0_vmax)
    sm = ScalarMappable(norm=norm, cmap='inferno')
    cbar = plt.colorbar(sm, cax=ax_cbar, orientation='vertical')
    cbar.ax.set_yticks([])
    cbar.ax.tick_params(size=0)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    fontsize = 12
    cbar.ax.text(0.5, 1.02, str(F_F0_vmax), ha='center', va='bottom', fontsize=fontsize, transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.02, str(F_F0_vmin), ha='center', va='top', fontsize=fontsize, transform=cbar.ax.transAxes)
    cbar.ax.text(1.3, 0.5, 'F/F0', ha='left', va='center', fontsize=fontsize, rotation=90, transform=cbar.ax.transAxes)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()

def process_and_plot_flim_images(filelist, power_slope, power_intercept, 
                               F_F0_vmin=1, F_F0_vmax=10,
                               result_img_dict=None,
                               from_Thorlab_to_coherent_factor=1/3,
                               plot_type='multiple'):
    """
    Process and plot FLIM images.
    
    Args:
        filelist: List of FLIM file paths
        power_slope: Slope of power calibration
        power_intercept: Intercept of power calibration
        F_F0_vmin: Minimum value for color scale
        F_F0_vmax: Maximum value for color scale
        result_img_dict: Dictionary to store results (optional)
        from_Thorlab_to_coherent_factor: Conversion factor
        plot_type: 'single' or 'multiple' plotting mode
    
    Returns:
        Dictionary containing processed image data
    """
    if result_img_dict is None:
        result_img_dict = {}
    
    for each_file in filelist: 
        print(f"Processing: {each_file}")
        if each_file in result_img_dict:
            continue
            
        data = process_flim_image(each_file, power_slope, power_intercept, 
                                from_Thorlab_to_coherent_factor)
        if data is not None:
            result_img_dict[each_file] = data

    if plot_type == 'single':
        # Plot each image individually
        for each_file, data in result_img_dict.items():
            folder = os.path.dirname(each_file)
            savefolder = os.path.join(folder, "plot")
            basename = os.path.basename(each_file)
            savepath = os.path.join(savefolder, basename[:-5] + ".png")
            plot_single_image(data, F_F0_vmin, F_F0_vmax, savepath)
    else:
        # Plot all images in one figure
        if result_img_dict:
            first_file = list(result_img_dict.keys())[0]
            folder = os.path.dirname(first_file)
            savefolder = os.path.join(folder, "plot")
            basename = os.path.basename(first_file)
            savepath = os.path.join(savefolder, basename[:-8] + ".png")
            plot_multiple_images(result_img_dict, F_F0_vmin, F_F0_vmax, savepath)
    
    return result_img_dict 