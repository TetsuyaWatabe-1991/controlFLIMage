# %%
import sys
sys.path.append(r"..")
import os
from FLIMageFileReader2 import FileReader
from FLIMageAlignment import get_xyz_pixel_um
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
binary_cmap = ListedColormap(['black', 'red'])
from scipy.ndimage import median_filter


def find_surface_1d(ZYXarray, threshold = 5):
    ZYXarray_denoised = median_filter(ZYXarray, size=(1, 5, 5))
    Z_intensity_max_1d = ZYXarray_denoised.max(axis=-1).max(axis=-1)
    binarized_1d = Z_intensity_max_1d > threshold

    if binarized_1d[-1]==True:
        print("Bright points found on the top of the image")  
        return len(binarized_1d) - 1
    else:
        for i in range(1, len(binarized_1d) - 1):
            if not binarized_1d[-i] and binarized_1d[-i - 1]:
                return len(binarized_1d) - i - 1
        else:
            print("No surface found")
            return len(binarized_1d) - 1


if __name__ == "__main__":

    flim_path_list = [
        r"G:\ImagingData\Tetsuya\20250612\test_lowmag\lowmag1_001_20250602_256_zoom2_GFP.flim",
        r"G:\ImagingData\Tetsuya\20250612\test_lowmag\lowmag1_001_20250523_auto2_256_zoom2_tdTom.flim",
        r"G:\ImagingData\Tetsuya\20250612\test_lowmag\lowmag1_001_20250601_4lines3auto_256_zoom1_GFP.flim",
        r"G:\ImagingData\Tetsuya\20250612\test_lowmag\lowmag1_001_20250612_512_zoom3_tdTom.flim"
        ]

    threshold = 5
    plt_savefolder = r"G:\ImagingData\Tetsuya\20250612\test_lowmag\pltsave"

    for flim_path in flim_path_list:
        iminfo = FileReader()
        iminfo.read_imageFile(flim_path, True)
        x_um, _, z_um = get_xyz_pixel_um(iminfo)

        ZYXarray = np.array(iminfo.image).sum(axis=tuple([1,2,5]))
        ZYXarray_denoised = median_filter(ZYXarray, size=(1, 5, 5))
        Z_intensity_max_1d = ZYXarray_denoised.max(axis=-1).max(axis=-1)
        Z_intensity_mean_1d = ZYXarray_denoised.mean(axis=-1).mean(axis=-1)

        surface_idx = find_surface_1d(ZYXarray, threshold = threshold)
        print(surface_idx)

        plt.subplot(2,3,1)
        plt.imshow(ZYXarray.max(axis=0), cmap="gray")
        plt.subplot(2,3,2)
        plt.plot(Z_intensity_max_1d, label="max",color="r")
        plt.axhline(threshold, color="k", linestyle="--")
        plt.axvline(surface_idx, color="k", linestyle="--")
        plt.ylabel("Max Intensity")
        plt.xlabel("Z")

        plt.subplot(2,3,3)
        plt.plot(Z_intensity_mean_1d, label="mean",color="b")    
        plt.axvline(surface_idx, color="k", linestyle="--")
        plt.ylabel("Mean Intensity")
        plt.xlabel("Z")

        plt.subplot(2,3,4)
        if surface_idx < ZYXarray.shape[0] - 1:
            plt.imshow(ZYXarray[surface_idx+1, :, :], cmap="gray")
        else:
            plt.imshow(np.zeros_like(ZYXarray[surface_idx, :, :]), cmap="gray")
            # plot a line from the top left to the bottom right
            plt.plot([0, ZYXarray.shape[1]-1], [0, ZYXarray.shape[2]-1], color="r", linewidth=2)
        plt.title("Above")
        plt.subplot(2,3,5)
        plt.imshow(ZYXarray[surface_idx, :, :], cmap="gray")
        plt.title("Surface")
        plt.subplot(2,3,6)
        plt.imshow(ZYXarray[surface_idx-1, :, :], cmap="gray")
        plt.title("Below")
        plt.tight_layout()
        plt.savefig(os.path.join(plt_savefolder, f"{os.path.basename(flim_path)}_surface_detection.png"))
        plt.show()
        

        
        
# %%
