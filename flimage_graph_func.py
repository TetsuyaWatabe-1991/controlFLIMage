# -*- coding: utf-8 -*-
"""
Created on Fri May  9 16:09:00 2025

@author: yasudalab
"""
import json
import sys
sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")
import os
import numpy as np
from custom_plot import plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from skimage.draw import disk, polygon
from scipy.ndimage import median_filter
from FLIMageFileReader2 import FileReader
from multidim_tiff_viewer import read_xyz_single
import pandas as pd
plt.rcParams['image.interpolation'] = 'none'

def plot_drift_from_drift_txt(drift_txt_path, show = True, savefig = False, savepath = ""):
    if False:
        drift_txt_path = r"G:\ImagingData\Tetsuya\20250611\test2\lowmag1_001_highmag_001_drift.txt"
    df = pd.read_csv(drift_txt_path, sep = ",")

    plt.figure(figsize=(4, 3))
    # Change order to plot z last so it's on top
    label_list=["drift_x_um","drift_y_um","drift_z_um"]  # Changed order
    col_dict={"drift_x_um":"g","drift_y_um":"k","drift_z_um":"m"}
    marker_dict = {"drift_x_um":"$x$", "drift_y_um":"$y$", "drift_z_um":"$z$"} 
    size_dict = {"drift_x_um":50, "drift_y_um":50, "drift_z_um":50}  # Increased sizes, z is largest
    filenum_list = range(len(df))
    zyx_drift_dict = {}
    for xyz in label_list:
        if xyz not in df.columns:
            if " "+xyz in df.columns:
                zyx_drift_dict[xyz] = df[" "+xyz]
            else:
                print(f"Error: {xyz} not found in {drift_txt_path}")
                return None
        else:
            zyx_drift_dict[xyz] = df[xyz]
    for xyz in label_list:
        if zyx_drift_dict[xyz].dtype != "float64":
            print(f"Error: {xyz} is not a float64 in {drift_txt_path}")
            return None

    for xyz in label_list:
        plt.plot(filenum_list, zyx_drift_dict[xyz], c=col_dict[xyz], ls="-", label=xyz)
    # Plot scatter points in reverse order (z last so it's on top)
    for xyz in reversed(label_list):
        plt.scatter(filenum_list, zyx_drift_dict[xyz], 
                    c=col_dict[xyz], 
                    marker=marker_dict[xyz],
                    s=size_dict[xyz],
                    alpha=0.8)  # slightly less transparent
    plt.xlabel("order of acquisition")
    plt.ylabel("\u03BCm")
    
    if savefig:
        if savepath == "":
            savepath = os.path.join(os.path.dirname(drift_txt_path), "drift_plot.png")
        plt.savefig(savepath,dpi=300,bbox_inches='tight')        
    if show==True:
        plt.show()        
    plt.close();plt.clf();plt.close("all");        


def color_fue(savefolder = r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage\ForUse",
              vmin =1, vmax=10, cmap='inferno', label_text = "F/F0",fontsize = 48,
              savefig = True
              ):
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    
    # ===== Vertical Right =====
    fig, ax = plt.subplots(figsize=(2, 6), facecolor='none')
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical')
    cbar.ax.set_yticks([])
    cbar.ax.tick_params(size=0)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    
    # Add value labels
    cbar.ax.text(0.5, 1.02, str(vmax), ha='center', va='bottom', fontsize=fontsize, transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.02, str(vmin), ha='center', va='top', fontsize=fontsize, transform=cbar.ax.transAxes)
    
    # Add F/F₀ label to right side
    cbar.ax.text(1.3, 0.5, label_text, ha='left', va='center', fontsize=fontsize, rotation=90, transform=cbar.ax.transAxes)
    
    plt.tight_layout()
    savepath = os.path.join(savefolder, f"vert_rt_{vmin}to{vmax}.png")
    if savefig:
        plt.savefig(savepath, dpi = 150, bbox_inches = "tight")
        plt.close(); plt.clf();
    else:
        plt.show()
        
    # ===== Vertical Left =====
    fig, ax = plt.subplots(figsize=(2, 6), facecolor='none')
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical')
    cbar.ax.set_yticks([])
    cbar.ax.tick_params(size=0)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    
    # Value labels
    cbar.ax.text(0.5, 1.02, str(vmax), ha='center', va='bottom', fontsize=fontsize, transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.02, str(vmin), ha='center', va='top', fontsize=fontsize, transform=cbar.ax.transAxes)
    
    # Label on left side
    cbar.ax.text(-0.3, 0.5, label_text, ha='right', va='center', fontsize=fontsize, rotation=90, transform=cbar.ax.transAxes)
    
    plt.tight_layout()
    savepath = os.path.join(savefolder, f"vert_lt_{vmin}to{vmax}.png")
    if savefig:
        plt.savefig(savepath, dpi = 150, bbox_inches = "tight")
        plt.close(); plt.clf();
    else:
        plt.show()
    
    
    
    
    # ===== Horizontal =====
    fig, ax = plt.subplots(figsize=(6, 2), facecolor='none')
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.ax.set_xticks([])
    cbar.ax.tick_params(size=0)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    
    # Value labels
    cbar.ax.text(-0.02, 0.5, str(vmin), ha='right', va='center', fontsize=fontsize, transform=cbar.ax.transAxes)
    cbar.ax.text(1.02, 0.5, str(vmax), ha='left', va='center', fontsize=fontsize, transform=cbar.ax.transAxes)
    
    # Label below bar
    cbar.ax.text(0.5, -0.1, label_text, ha='center', va='top', fontsize=fontsize, transform=cbar.ax.transAxes)
    
    plt.tight_layout()
    savepath = os.path.join(savefolder, f"hori_{vmin}to{vmax}.png")
    if savefig:
        plt.savefig(savepath, dpi = 150, bbox_inches = "tight")
        plt.close(); plt.clf();
    else:
        plt.show()


def get_nice_scalebar_um(image_x_um):
    """Return a nice integer scalebar length approximately 1/10 to 1/5 of image_x_um."""
    target = image_x_um * 0.15
    magnitude = 10 ** np.floor(np.log10(target))
    candidates = [magnitude, 2 * magnitude, 5 * magnitude, 10 * magnitude]
    return int(min(candidates, key=lambda v: abs(v - target)))


def add_scale_bar(ax, scalebar_um, x_um_per_pix, avoid_x_y=None, color='white', fontsize=10):
    """Draw a scale bar with label in the upper right corner of the axes.

    If avoid_x_y is provided and falls within the upper right region,
    the bar is placed in the upper left instead.

    Args:
        ax: matplotlib Axes object
        scalebar_um: scale bar length in micrometers
        x_um_per_pix: micrometers per pixel in x direction
        avoid_x_y: (x, y) data coordinate to avoid overlapping (e.g. uncaging position)
        color: bar and text color
        fontsize: font size of the label
    """
    scalebar_pix = scalebar_um / x_um_per_pix
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = abs(xlim[1] - xlim[0])
    y_range = abs(ylim[1] - ylim[0])
    margin_x = x_range * 0.03
    margin_y = y_range * 0.05

    # Default: upper right
    x_right = max(xlim) - margin_x
    x_left = x_right - scalebar_pix
    # imshow has inverted y-axis: min(ylim) is the visual top
    y_bar = min(ylim) + margin_y
    y_text = y_bar - y_range * 0.015

    if avoid_x_y is not None:
        avoid_x, avoid_y = avoid_x_y
        # Check if avoid point falls in the upper right region (with generous padding)
        in_right_half = avoid_x >= min(xlim) + x_range * 0.5
        in_upper_band = avoid_y <= min(ylim) + y_range * 0.2
        if in_right_half and in_upper_band:
            # Switch to upper left
            x_left = min(xlim) + margin_x
            x_right = x_left + scalebar_pix

    ax.plot([x_left, x_right], [y_bar, y_bar],
            color=color, linewidth=2, solid_capstyle='butt')
    ax.text((x_left + x_right) / 2, y_text, f'{scalebar_um} \u03bcm',
            color=color, ha='center', va='bottom', fontsize=fontsize)


def plot_max_proj_uncaging(
                    each_file, ch_1or2, show_uncaging = False,
                    use_default_savefolder = True, savefolder = "",
                    plot_text = "",
                    give_uncaging_x_y_0to1 = False,
                    give_center_x_y = False,
                    defined_center_x_y = [0, 0],
                    defined_uncaging_x_y_0to1 = [0, 0],
                    vmin = 0, vmax = 0,
                    return_vmax = False,
                    marker = 'c.'
                    ):
    iminfo = FileReader()
    iminfo.read_imageFile(each_file, True) 
    x_um_per_pix_x = iminfo.statedict['State.Acq.FOV_default'][0] / iminfo.statedict['State.Acq.zoom'] / iminfo.statedict['State.Acq.pixelsPerLine']
    ch = ch_1or2 - 1
    imagearray=np.array(iminfo.image)
    
    if give_uncaging_x_y_0to1:
        uncaging_x_y_0to1 = defined_uncaging_x_y_0to1
    else:
        uncaging_x_y_0to1 = iminfo.statedict["State.Uncaging.Position"]

    if give_center_x_y:
        center_x = defined_center_x_y[0]
        center_y = defined_center_x_y[1]
    else:
        center_y = imagearray.shape[-2] * uncaging_x_y_0to1[1]
        center_x = imagearray.shape[-3] * uncaging_x_y_0to1[0]
    
    maxproj = imagearray[:, 0, ch, :,:,:].sum(axis=-1).sum(axis=0)
      
    plt.figure(facecolor='black')
    if vmax == 0:
        vmax = maxproj.max() * 0.7
    plt.imshow(maxproj, cmap = 'gray', vmin = 0, vmax = vmax)
    if show_uncaging:
        plt.plot(center_x, center_y, marker, markersize=10)   
        folder = os.path.dirname(each_file)
        savefolder = os.path.join(folder,"plot_maxproj")
        trimmed_savefolder = os.path.join(folder,"plot_maxproj_trimmed")
    else:
        savefolder = savefolder
        trimmed_savefolder = os.path.join(savefolder,"plot_maxproj_trimmed")
    os.makedirs(savefolder, exist_ok=True)
    os.makedirs(trimmed_savefolder, exist_ok=True)
    plt.axis('off')
    if plot_text != "":
        if type(plot_text) == str:
            #plot text at the left top corner of the plot
            # plt.text(0.02, 0.99, plot_text, ha='left', va='top', fontsize=14, transform=plt.gca().transAxes,
            # color='yellow')
            plt.title(plot_text, color='yellow')
    image_x_um = maxproj.shape[1] * x_um_per_pix_x
    scalebar_um = get_nice_scalebar_um(image_x_um)
    add_scale_bar(plt.gca(), scalebar_um, x_um_per_pix_x, avoid_x_y=(center_x, center_y))
    basename = os.path.basename(each_file)
    savepath = os.path.join(savefolder, basename[:-5] + "_maxproj.png")
    
    plt.savefig(savepath, dpi=150, bbox_inches = "tight", pad_inches = 0)
    print("maxproj_savepath ", savepath)
    plt.show()

    trim_side_length = 25
    # ylim = [min(maxproj.shape[0], center_y + trim_side_length),max(0, center_y - trim_side_length)]
    # xlim = [max(0, center_x - trim_side_length), min(maxproj.shape[1], center_x + trim_side_length)]

    ylim = [center_y + trim_side_length, center_y - trim_side_length]
    xlim = [center_x - trim_side_length, center_x + trim_side_length]


    if show_uncaging:
        plt.figure(facecolor='black')
        plt.imshow(maxproj, cmap = 'gray', vmin = 0, vmax = vmax)
        plt.plot(center_x, center_y, marker, markersize=20)   
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.axis('off')
        plt.title(plot_text, color='yellow')
        # plt.text(0.02, 0.99, plot_text, ha='left', va='top', fontsize=14, transform=plt.gca().transAxes,
        #     color='yellow')
        trimmed_x_um = 2 * trim_side_length * x_um_per_pix_x
        trim_scalebar_um = get_nice_scalebar_um(trimmed_x_um)
        add_scale_bar(plt.gca(), trim_scalebar_um, x_um_per_pix_x, avoid_x_y=(center_x, center_y))
        trimmed_savepath = os.path.join(trimmed_savefolder, basename[:-5] + "_maxproj_trimmed.png")
        plt.savefig(trimmed_savepath, dpi=150, bbox_inches = "tight", pad_inches = 0)
        print("trimmed_maxproj_savepath ", trimmed_savepath)
        plt.show()


def plot_GCaMP_F_F0(each_file, slope = 0, intercept = 0, 
                    from_Thorlab_to_coherent_factor = 1/3,
                    vmin = 1, vmax = 10, cmap='inferno', 
                    acceptable_image_shape_0th_list = [4,32, 33,34,55],
                    GCaMP_intensity_threshold = 0,
                    plot_RFP_also = False):
    uncaging_iminfo = FileReader()
    uncaging_iminfo.read_imageFile(each_file, True) 
    
    imagearray=np.array(uncaging_iminfo.image)
    
    if (imagearray.shape)[0] not in acceptable_image_shape_0th_list:
        print("Image shape is not expected size.  ",imagearray.shape)
        return
    
    uncaging_x_y_0to1 = uncaging_iminfo.statedict["State.Uncaging.Position"]
    uncaging_pow = uncaging_iminfo.statedict["State.Uncaging.Power"]
    pulseWidth = uncaging_iminfo.statedict["State.Uncaging.pulseWidth"]
    center_y = imagearray.shape[-2] * uncaging_x_y_0to1[1]
    center_x = imagearray.shape[-3] * uncaging_x_y_0to1[0]
    
    if imagearray.shape[0] in [4, 33, 34]:
        GCpre = imagearray[1,0,0,:,:,:].sum(axis=-1)
        GCunc = imagearray[2,0,0,:,:,:].sum(axis=-1)
        RFPpre = imagearray[0, 0, 1, :, :, :].sum(axis=-1)
    elif imagearray.shape[0] in [32]:
        GCpre = imagearray[8*1 + 1 : 8*2, 0,0,:,:,:].sum(axis=-1).sum(axis=0)
        GCunc = imagearray[8*2 + 1 : 8*3, 0,0,:,:,:].sum(axis=-1).sum(axis=0)
        RFPpre = imagearray[8*1 + 1 : 8*2, 0, 1, :, :, :].sum(axis=-1).sum(axis=0)
    elif imagearray.shape[0] in [55]:
        GCpre = imagearray[4,0,0,:,:,:].sum(axis=-1)
        GCunc = imagearray[5,0,0,:,:,:].sum(axis=-1)
    assert len(GCpre.shape) == 2 #Image should be 2D



    GC_pre_med = median_filter(GCpre, size=3)
    GC_unc_med = median_filter(GCunc, size=3)
    
    GCF_F0 = (GC_unc_med/GC_pre_med)
    GCF_F0[GC_pre_med <= GCaMP_intensity_threshold] = 0
       
    pow_mw = slope * uncaging_pow + intercept
    pow_mw_coherent = pow_mw*from_Thorlab_to_coherent_factor
    pow_mw_round = round(pow_mw_coherent,1)
       
    plt.imshow(GCF_F0, cmap = cmap, vmin = vmin, vmax = vmax)
    plt.plot(center_x, center_y, 'co', markersize=4)   
    
    if pow_mw_round > 0:
        plt.title(f"{pow_mw_round} mW, {pulseWidth} ms")  
    else:
        plt.title(f"{uncaging_pow} %, {pulseWidth} ms")  
    plt.axis('off')
    
    folder = os.path.dirname(each_file)
    savefolder = os.path.join(folder,"plot")
    os.makedirs(savefolder, exist_ok=True)
    basename = os.path.basename(each_file)                
    savepath = os.path.join(savefolder, basename[:-5] + "_F_F0.png")
    plt.savefig(savepath, dpi=150, bbox_inches = "tight")
    print("F_F0_savepath ", savepath)
    plt.show()
    plt.close(); plt.clf();plt.close("all");
    
    color_fue(savefolder = savefolder,
              vmin =vmin, vmax=vmax, cmap=cmap, label_text = "F/F0")
    
    if plot_RFP_also:        
        plt.imshow(RFPpre, cmap='gray', vmin=0)
        plt.plot(center_x, center_y, 'co', markersize=4)
        plt.title("RFP")
        plt.axis('off')
        savepath = os.path.join(savefolder, basename[:-5] + "_RFP.png")
        plt.savefig(savepath, dpi=150, bbox_inches = "tight")
        print("RFP_savepath ", savepath)
        plt.show()
        plt.close(); plt.clf();plt.close("all");


def plot_GCaMP_and_RFP(each_file, slope = 0, intercept = 0, 
                    from_Thorlab_to_coherent_factor = 1/3,
                    vmin = 1, vmax = 10, cmap='inferno', 
                    acceptable_image_shape_0th_list = [4,32, 33,34],
                    ch1or2 = 2):
    """
    Plot GCaMP F/F0 and RFP (ch1or2=2) first frame side by side
    Left: GCaMP F/F0
    Right: RFP (ch1or2=2) first frame (with uncaging pos)
    """
    uncaging_iminfo = FileReader()
    uncaging_iminfo.read_imageFile(each_file, True) 
    
    imagearray=np.array(uncaging_iminfo.image)
    
    if (imagearray.shape)[0] not in acceptable_image_shape_0th_list:
        print("Image shape is not expected size.  ",imagearray.shape)
        return
    
    uncaging_x_y_0to1 = uncaging_iminfo.statedict["State.Uncaging.Position"]
    uncaging_pow = uncaging_iminfo.statedict["State.Uncaging.Power"]
    pulseWidth = uncaging_iminfo.statedict["State.Uncaging.pulseWidth"]
    center_y = imagearray.shape[-2] * uncaging_x_y_0to1[1]
    center_x = imagearray.shape[-3] * uncaging_x_y_0to1[0]
    
    # Calculate GCaMP F/F0 (for left plot)
    if imagearray.shape[0] in [4, 33, 34]:
        GCpre = imagearray[1,0,0,:,:,:].sum(axis=-1)
        GCunc = imagearray[2,0,0,:,:,:].sum(axis=-1)
    elif imagearray.shape[0] in [32]:
        GCpre = imagearray[8*1 + 1 : 8*2, 0,0,:,:,:].sum(axis=-1).sum(axis=0)
        GCunc = imagearray[8*2 + 1 : 8*3, 0,0,:,:,:].sum(axis=-1).sum(axis=0)
    assert len(GCpre.shape) == 2 #Image should be 2D

    GC_pre_med = median_filter(GCpre, size=3)
    GC_unc_med = median_filter(GCunc, size=3)
    
    GCF_F0 = (GC_unc_med/GC_pre_med)
    GCF_F0[GC_pre_med == 0] = 0
    
    # Get RFP (ch1or2=2) first frame
    ch = ch1or2 - 1  # ch1or2=2 -> ch=1
    if imagearray.shape[0] in [4, 33, 34]:
        RFP_image = imagearray[0, 0, ch, :, :, :].sum(axis=-1)
    elif imagearray.shape[0] in [32]:
        RFP_image = imagearray[8*0 + 1 : 8*1, 0, ch, :, :, :].sum(axis=-1).sum(axis=0)
    assert len(RFP_image.shape) == 2 #Image should be 2D
    
    pow_mw = slope * uncaging_pow + intercept
    pow_mw_coherent = pow_mw/3
    pow_mw_round = round(pow_mw_coherent,1)
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Set overall title with file path
    fig.suptitle(each_file, fontsize=10, y=0.98)
    
    # Left: GCaMP F/F0
    ax1.imshow(GCF_F0, cmap = cmap, vmin = vmin, vmax = vmax)
    ax1.plot(center_x, center_y, 'co', markersize=4)
    if pow_mw_round > 0:
        ax1.set_title(f"GCaMP F/F0\n{pow_mw_round} mW, {pulseWidth} ms")
    else:
        ax1.set_title(f"GCaMP F/F0\n{uncaging_pow} %, {pulseWidth} ms")
    ax1.axis('off')
    
    # Right: RFP (ch1or2=2) first frame
    ax2.imshow(RFP_image, cmap='gray', vmin=0)
    ax2.plot(center_x, center_y, 'co', markersize=4)
    ax2.set_title(f"RFP (ch{ch1or2}) - 1st frame")
    ax2.axis('off')
    
    plt.tight_layout()
    
    folder = os.path.dirname(each_file)
    savefolder = os.path.join(folder, "plot_GCaMP_and_RFP")
    os.makedirs(savefolder, exist_ok=True)
    basename = os.path.basename(each_file)
    savepath = os.path.join(savefolder, basename[:-5] + "_GCaMP_and_RFP.png")
    plt.savefig(savepath, dpi=150, bbox_inches = "tight")
    print("GCaMP_and_RFP_savepath ", savepath)
    plt.show()
    plt.close(); plt.clf();plt.close("all");
    
    color_fue(savefolder = savefolder,
              vmin =vmin, vmax=vmax, cmap=cmap, label_text = "F/F0")

    
    
def calc_point_on_line_close_to_xy(x, y, slope, intercept):   
    x_c = (x + slope * (y - intercept)) / (slope**2 + 1)
    y_c = slope * x_c + intercept
    return y_c, x_c


def calc_spine_dend_GCaMP(
    each_file,
    spine_y = -1,
    spine_x = -1,    
    drift_y_pix = 0,
    drift_x_pix = 0,
    dend_slope = 0,
    dend_intercept = 0,
    each_ini = "",
    from_ini = True,
    circle_radius = 3,  # Set as needed
    rect_length = 10,  # along the line
    rect_height = 2,   # perpendicular to the line
    save_img = False,
    save_suffix = "",
    ):
    
    if from_ini:
        spine_zyx, dend_slope, dend_intercept, excluded = read_xyz_single(each_ini,
                                                          return_excluded = True)
        spine_x = spine_zyx[2]
        spine_y = spine_zyx[1]
    
    
    #drift correction
    spine_x -= drift_x_pix
    spine_y -= drift_y_pix
    dend_intercept -=  -dend_slope * drift_x_pix + drift_y_pix
    # spine_x += drift_x_pix
    # spine_y += drift_y_pix
    # dend_intercept +=  -dend_slope * drift_x_pix + drift_y_pix
    
    y_c, x_c = calc_point_on_line_close_to_xy(x = spine_x, y = spine_y, 
                                   slope = dend_slope, 
                                   intercept = dend_intercept)
    
    
    
    uncaging_iminfo = FileReader()
    uncaging_iminfo.read_imageFile(each_file, True) 
    
    imagearray=np.array(uncaging_iminfo.image)
    
    if imagearray.shape[0] in [4, 33, 34]:
        GCpre = imagearray[1,0,0,:,:,:].sum(axis=-1)
        GCunc = imagearray[2,0,0,:,:,:].sum(axis=-1)
    elif imagearray.shape[0] in [32]:
        GCpre = imagearray[8*1 + 1 : 8*2, 0,0,:,:,:].sum(axis=-1).sum(axis=0)
        GCunc = imagearray[8*2 + 1 : 8*3, 0,0,:,:,:].sum(axis=-1).sum(axis=0)
    else:
        print("Image shape is not expected size.  ",imagearray.shape)
        return -1, -1
    assert len(GCpre.shape) == 2 #Image should be 2D

    GC_pre_med = median_filter(GCpre, size=3)
    GC_unc_med = median_filter(GCunc, size=3)
    
    GCF_F0 = (GC_unc_med/GC_pre_med)
    GCF_F0[GC_pre_med == 0] = 0
    # print(f"Closest point on the line: ({x_c:.3f}, {y_c:.3f})")
      
    ### circle
    
    rr_circ, cc_circ = disk((spine_y, spine_x), circle_radius, shape=GC_unc_med.shape)
    spine_mean_pre = GC_pre_med[rr_circ, cc_circ].mean()
    spine_mean_unc = GC_unc_med[rr_circ, cc_circ].mean()
        
    ### rectangle
    
    # 2. Rectangle ROI aligned with the dendrite line
    theta = np.arctan(dend_slope)
    dx = (rect_length / 2) * np.cos(theta)
    dy = (rect_length / 2) * np.sin(theta)
    px = (rect_height / 2) * -np.sin(theta)
    py = (rect_height / 2) * np.cos(theta)
    
    # Rectangle corners
    corners_x = [x_c - dx - px, x_c - dx + px, x_c + dx + px, x_c + dx - px]
    corners_y = [y_c - dy - py, y_c - dy + py, y_c + dy + py, y_c + dy - py]
    
    rr_rect, cc_rect = polygon(corners_y, corners_x, shape=GC_unc_med.shape)
    shaft_mean_pre = GC_pre_med[rr_rect, cc_rect].mean()
    shaft_mean_unc = GC_unc_med[rr_rect, cc_rect].mean()
    
    spineF_F0 = spine_mean_unc/spine_mean_pre
    shaftF_F0 = shaft_mean_unc/shaft_mean_pre

    # Plot
    
    vmax=np.percentile(GCunc,99.9)
    GCunc8bit=(GCunc/vmax * 255).astype(np.uint8)
    GCunc8bit[GCunc>vmax] = 255
    GCunc_24bit = np.zeros((GC_unc_med.shape[0], GC_unc_med.shape[1], 3), dtype=np.uint8)
    
    GCunc_24bit[:,:,0]=GCunc8bit
    GCunc_24bit[:,:,1]=GCunc8bit
    GCunc_24bit[:,:,2]=GCunc8bit
    GCunc_24bit[rr_circ, cc_circ,0] = 255
    GCunc_24bit[rr_circ, cc_circ,1] = 0
    GCunc_24bit[rr_circ, cc_circ,2] = 0
    GCunc_24bit[rr_rect, cc_rect,0] = 0
    GCunc_24bit[rr_rect, cc_rect,1] = 0
    GCunc_24bit[rr_rect, cc_rect,2] = 255
    plt.imshow(GCunc_24bit)
    plt.title("ROIs on Image")
    plt.axis("off")
    folder = os.path.dirname(each_file)
    savefolder = os.path.join(folder,"plot")
    os.makedirs(savefolder, exist_ok=True)
    basename = os.path.basename(each_file)                
    savepath = os.path.join(savefolder, basename[:-5] + "_ROI" + save_suffix + ".png")
    if save_img:
        plt.savefig(savepath, dpi=150, bbox_inches = "tight")
    print("ROI_savepath ", savepath)
    plt.show()
    

    print("spine",round(spine_mean_pre,1), round(spine_mean_unc,1), " F_F0",round(spineF_F0,1))
    print("shaft",round(shaft_mean_pre,1), round(shaft_mean_unc,1), " F_F0",round(shaftF_F0,1))

    return spineF_F0, shaftF_F0
    
    
if __name__ == "__main__":
    each_file = r"C:\Users\yasudalab\Desktop\0319_25deep_2_pos1__highmag_4_007.flim"
    ch_1or2 = 2
    plot_max_proj_uncaging(
                    each_file, ch_1or2, show_uncaging = True,
                    use_default_savefolder = True, savefolder = "",
                    plot_text = "XX.X min after uncaging",
                    give_uncaging_x_y_0to1 = False,
                    give_center_x_y = False,
                    defined_center_x_y = [0, 0],
                    defined_uncaging_x_y_0to1 = [0, 0],
                    vmin = 0, vmax = 0,
                    return_vmax = False,
                    )