import os
import sys
sys.path.append(r"..")
from pathlib import Path
import matplotlib.pyplot as plt 
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.ndimage import median_filter, binary_dilation
from FLIMageFileReader2 import FileReader
from FLIMageAlignment import get_xyz_pixel_um
from utility.surface_detection import find_surface_1d
from FLIMageAlignment import get_xyz_pixel_um
from after_click_image_func import save_pix_pos_from_click_list, save_um_pos_from_click_list, get_abs_um_pos_from_center_3d


plt.rcParams['image.interpolation'] = 'none'

def filter_skeleton_points(skeleton, min_component_size=10, connection_radius=3):
    """
    Filter skeleton points based on component size and connection radius.
    
    Args:
        skeleton (numpy.ndarray): Binary skeleton image.
        min_component_size (int): Minimum size of valid components.
        connection_radius (int): Maximum distance between connected points.
    
    Returns:
        numpy.ndarray: Filtered skeleton points.
    """
    
    skeleton_points = np.array(np.where(skeleton)).T
    
    if len(skeleton_points) == 0:
        return skeleton_points
    
    tree = cKDTree(skeleton_points)
    pairs = tree.query_pairs(r=connection_radius)
    
    if len(pairs) > 0:
        n_points = len(skeleton_points)
        pairs_array = np.array(list(pairs))
        adjacency_matrix = csr_matrix((np.ones(len(pairs)), (pairs_array[:,0], pairs_array[:,1])), shape=(n_points, n_points))
        n_components, labels = connected_components(adjacency_matrix)

        component_sizes = np.bincount(labels)
        valid_components = np.where(component_sizes >= min_component_size)[0]
        valid_points = np.isin(labels, valid_components)
        skeleton_points_filtered = skeleton_points[valid_points]
        
        return skeleton_points_filtered
    else:
        return skeleton_points

def skeletonize_with_branch_filtering(binary_image, min_branch_length=10, min_component_size=10, connection_radius=3):
    from skimage.morphology import remove_small_objects, skeletonize_3d
    
    # Method 1: Remove small objects before skeletonization
    binary_cleaned = remove_small_objects(binary_image, min_size=min_component_size)
    
    # Method 2: Use 3D skeletonization with pruning
    skeleton_3d = skeletonize_3d(binary_cleaned)
    
    # Method 3: Apply our custom filtering
    skeleton_points_filtered = filter_skeleton_points(skeleton_3d, min_component_size, connection_radius)
    
    # Create final skeleton
    skeleton_final = np.zeros_like(skeleton_3d)
    if len(skeleton_points_filtered) > 0:
        skeleton_final[tuple(skeleton_points_filtered.T)] = True
    
    return skeleton_final, skeleton_points_filtered


def get_skeleton_3d(flim_path, min_dendrite_length_um, ch_specified, ch1or2,
                    percentile):
    iminfo = FileReader()
    iminfo.read_imageFile(flim_path, True)
    x_um, _, z_um = get_xyz_pixel_um(iminfo)
    min_dendrite_length_pixel = int(min_dendrite_length_um/x_um)
    if ch_specified:
        ZYXarray = (np.array(iminfo.image)[:,:,ch1or2 - 1,:,:,:]).sum(axis=tuple([1,4]))
    else:
        ZYXarray = np.array(iminfo.image).sum(axis=tuple([1,2,5])) 
    
    ZYX_denoised = median_filter(ZYXarray, size=(1, 3, 3))
    threshold_3d = np.percentile(ZYX_denoised, percentile)
    ZYX_denoised_bin = ZYX_denoised > threshold_3d
    skeleton_3d, skeleton_points_3d = skeletonize_with_branch_filtering(
        ZYX_denoised_bin, 
        min_branch_length=min_dendrite_length_pixel, 
        min_component_size=10, 
        connection_radius=2
    )
    return skeleton_3d, skeleton_points_3d

    
def shaft_plot_example():
    from matplotlib.colors import ListedColormap
    binary_cmap = ListedColormap(['black', 'red'])

    plt_savefolder = r"G:\ImagingData\Tetsuya\20250612\test_lowmag\skeleton_test"
    flim_path_list = [
        r"G:\ImagingData\Tetsuya\20250612\test_lowmag\lowmag1_001_20250602_256_zoom2_GFP.flim",
        r"G:\ImagingData\Tetsuya\20250612\test_lowmag\lowmag1_001_20250523_auto2_256_zoom2_tdTom.flim",
        r"G:\ImagingData\Tetsuya\20250612\test_lowmag\lowmag1_001_20250601_4lines3auto_256_zoom1_GFP.flim",
        r"G:\ImagingData\Tetsuya\20250612\test_lowmag\lowmag1_001_20250612_512_zoom3_tdTom.flim"
        ]
    ch_specified = False
    ch1or2 = 1
    min_dendrite_length_um = 25


    for flim_path in flim_path_list:
        iminfo = FileReader()
        iminfo.read_imageFile(flim_path, True)
        x_um, _, z_um = get_xyz_pixel_um(iminfo)
        min_dendrite_length_pixel = int(min_dendrite_length_um/x_um)
        if ch_specified:
            ZYXarray = (np.array(iminfo.image)[:,:,ch1or2 - 1,:,:,:]).sum(axis=tuple([1,4]))
        else:
            ZYXarray = np.array(iminfo.image).sum(axis=tuple([1,2,5]))

        plt.imshow(ZYXarray.max(axis=0), cmap="gray")
        plt.savefig(os.path.join(plt_savefolder, f"{os.path.basename(flim_path)}_ZYXarray.png"), 
                    dpi=300, bbox_inches = 'tight')
        plt.show()

        ZYX_denoised = median_filter(ZYXarray, size=(1, 3, 3))
        plt.imshow(ZYX_denoised.max(axis=0), cmap="gray")
        plt.savefig(os.path.join(plt_savefolder, f"{os.path.basename(flim_path)}_ZYX_denoised.png"), 
                    dpi=300, bbox_inches = 'tight')
        plt.show()
        
        plt.hist(ZYX_denoised.flatten(), bins=np.linspace(1, 10, 20), alpha=0.7)
        #show in log scale y
        # plt.yscale('log')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.title('Intensity Histogram of Denoised ZYX Array (1-10)')
        plt.savefig(os.path.join(plt_savefolder, f"{os.path.basename(flim_path)}_ZYX_denoised_hist.png"), 
                    dpi=300, bbox_inches = 'tight')
        plt.show()
 

        percentile_list = [98]
        font_size = 20
        for percentile in percentile_list:
            threshold_3d = np.percentile(ZYX_denoised, percentile)
            ZYX_denoised_bin = ZYX_denoised > threshold_3d
            plt.imshow(ZYX_denoised_bin.max(axis=0), cmap="gray")
            plt.text(0.02,0.02,f"percentile: {percentile}", fontsize=font_size,ha="left",
                     va="top",color="white")

            plt.savefig(os.path.join(plt_savefolder, f"{os.path.basename(flim_path)}_ZYX_denoised_bin_{threshold_3d}.png"), 
                        dpi=300, bbox_inches = 'tight')
            plt.show()

            skeleton_3d, skeleton_points_3d = get_skeleton_3d(
                flim_path, min_dendrite_length_um, ch_specified, ch1or2, percentile
            )
            
            plt.imshow(skeleton_3d.max(axis=0), cmap="gray")
            plt.text(0.02,0.02,f"percentile: {percentile}", fontsize=font_size,ha="left",
                     va="top",color="white")
           
            plt.savefig(os.path.join(plt_savefolder, f"{os.path.basename(flim_path)}_skeleton_3d_{percentile}.png"), 
                        dpi=300, bbox_inches = 'tight')
             
            plt.show()

            plt.imshow(ZYXarray.max(axis=0), cmap="gray")
            plt.imshow(skeleton_3d.max(axis=0), cmap=binary_cmap, alpha=0.3)
            plt.text(0.02,0.02,f"percentile: {percentile}", fontsize=font_size,ha="left",
                     va="top",color="white")

            plt.savefig(os.path.join(plt_savefolder, f"{os.path.basename(flim_path)}_skeleton_3d_filtered_ZYX_{percentile}.png"), 
                        dpi=300, bbox_inches = 'tight')
            plt.show()

def plot_skeleton_3d(skeleton_3d, x_um, y_um, z_um,
                     saveplot = False, savepath = None):
    YX_skeleton_3d = skeleton_3d.max(axis=0)
    ZX_skeleton_3d = skeleton_3d.max(axis=1)
    ZY_skeleton_3d = skeleton_3d.max(axis=2)
    
    single_side_length = 5
    plt.figure(figsize=(single_side_length, single_side_length*3))
    plt.subplot(1,3,1)
    plt.imshow(YX_skeleton_3d, cmap="gray",
               
               extent=[0, YX_skeleton_3d.shape[1]*x_um,0, YX_skeleton_3d.shape[0]*y_um])   
    plt.xlabel("X (um)")
    plt.ylabel("Y (um)")

    plt.subplot(1,3,2)
    plt.imshow(ZX_skeleton_3d, cmap="gray",               
               extent=[0, ZX_skeleton_3d.shape[1]*x_um, 0, ZX_skeleton_3d.shape[0]*z_um ])
    plt.xlabel("X (um)")
    plt.ylabel("Z (um)")
    
    plt.subplot(1,3,3)
    plt.imshow(ZY_skeleton_3d, cmap="gray",               
               extent=[0, ZY_skeleton_3d.shape[1]*y_um, 0, ZY_skeleton_3d.shape[0]*z_um])
    plt.xlabel("Y (um)")
    plt.ylabel("Z (um)")
    
    if saveplot:
        if savepath is not None:
            plt.savefig(savepath, dpi=300, bbox_inches = 'tight')
        else:
            print("savepath is not specified")
    plt.show()

def define_skeleton_points(skeleton_3d, spacing_um, x_um, y_um, z_um):
    """
    Define points on skeleton with specified spacing accounting for different pixel sizes.
    
    Args:
        skeleton_3d (numpy.ndarray): 3D binary skeleton
        spacing_um (float): Distance between points in microns
        x_um (float): Pixel size in x direction (microns)
        y_um (float): Pixel size in y direction (microns)
        z_um (float): Pixel size in z direction (microns)
    
    Returns:
        numpy.ndarray: Array of point coordinates (N, 3)
    """
    from scipy.spatial import cKDTree
    
    # Get all skeleton points
    skeleton_points = np.array(np.where(skeleton_3d)).T
    
    if len(skeleton_points) == 0:
        return np.array([])
    
    # Convert to physical coordinates (microns)
    physical_points = skeleton_points * np.array([z_um, y_um, x_um])
    
    # Start with first point
    selected_points = [physical_points[0]]
    remaining_points = physical_points[1:]
    
    while len(remaining_points) > 0:
        # Find closest point to last selected point
        tree = cKDTree(remaining_points)
        distances, indices = tree.query(selected_points[-1], k=1)
        
        # Handle case where k=1 returns scalar instead of array
        if np.isscalar(distances):
            distance = distances
            index = indices
        else:
            distance = distances[0]
            index = indices[0]
        
        if distance >= spacing_um:
            # Add this point if it's far enough
            selected_points.append(remaining_points[index])
        
        # Remove the closest point from remaining
        remaining_points = np.delete(remaining_points, index, axis=0)
    
    # Convert back to pixel coordinates
    selected_points_pixels = np.array(selected_points) / np.array([z_um, y_um, x_um])
    
    return selected_points_pixels.astype(int)


def plot_skeleton_with_points(skeleton_3d, skeleton_points,change_aspect_ratio=False,
                             saveplot=False, savepath=None):

    if change_aspect_ratio == False:
        aspect_ratio = 1
    else:
        aspect_ratio = skeleton_3d.shape[1]/skeleton_3d.shape[0]
        print(f"aspect_ratio: {aspect_ratio}")

    YX_skeleton_3d = skeleton_3d.max(axis=0)
    ZX_skeleton_3d = skeleton_3d.max(axis=1)
    ZY_skeleton_3d = skeleton_3d.max(axis=2)
    
    single_side_length = 5
    plt.figure(figsize=(single_side_length, single_side_length*3))
    
    scatter_size = 5
    # XY projection
    plt.subplot(1,3,1)
    plt.imshow(YX_skeleton_3d, cmap="gray")
    if len(skeleton_points) > 0:
        plt.scatter(skeleton_points[:, 2], skeleton_points[:, 1], 
                   c='red', s=scatter_size, alpha=0.8)
    plt.title("Z Projection")

    # ZX projection
    plt.subplot(1,3,2)
    plt.imshow(ZX_skeleton_3d, cmap="gray",aspect=aspect_ratio)
    if len(skeleton_points) > 0:
        plt.scatter(skeleton_points[:, 2], skeleton_points[:, 0], 
                   c='red', s=scatter_size, alpha=0.8)
    plt.title("Y Projection")
    
    # ZY projection
    plt.subplot(1,3,3)
    plt.imshow(ZY_skeleton_3d, cmap="gray",aspect=aspect_ratio)
    if len(skeleton_points) > 0:
        plt.scatter(skeleton_points[:, 1], skeleton_points[:, 0], 
                   c='red', s=scatter_size, alpha=0.8)
    plt.title("X Projection")
    
    plt.tight_layout()
    
    if saveplot:
        if savepath is not None:
            plt.savefig(savepath, dpi=300, bbox_inches='tight')
        else:
            print("savepath is not specified")
    plt.show()


def detect_and_remove_soma_from_original(ZYXarray, percentile, max_soma_size_um, x_um, y_um, z_um):
    """
    Detect soma-like structures from original ZYXarray and create mask to exclude them.
    
    Args:
        ZYXarray (numpy.ndarray): Original 3D image data
        percentile (float): Percentile for thresholding
        max_soma_size_um (float): Maximum soma diameter in microns
        x_um, y_um, z_um (float): Pixel sizes in microns
    
    Returns:
        numpy.ndarray: Binary mask with soma regions excluded
    """
    from scipy.ndimage import median_filter, binary_opening, binary_closing, label
    from skimage.morphology import remove_small_objects
    
    # Denoise the original array
    ZYX_denoised = median_filter(ZYXarray, size=(1, 3, 3))
    
    # Threshold to get binary image
    threshold_3d = np.percentile(ZYX_denoised, percentile)
    ZYX_binary = ZYX_denoised > threshold_3d
    
    # Calculate soma size in pixels
    max_soma_size_pixels = int(max_soma_size_um / min(x_um, y_um, z_um))
    
    # Remove small objects (noise) and keep large objects (potential soma)
    soma_mask = remove_small_objects(ZYX_binary, min_size=max_soma_size_pixels)
    
    # Apply morphological operations to smooth soma regions
    soma_mask = binary_opening(soma_mask, structure=np.ones((3,3,3)))
    soma_mask = binary_closing(soma_mask, structure=np.ones((5,5,5)))
    
    # Create mask excluding soma regions
    dendrite_mask = ZYX_binary & (~soma_mask)
    
    return dendrite_mask, soma_mask

def generate_skeleton_from_filtered_image(ZYXarray, threshold_area_um2, x_um, y_um, z_um, 
                                         soma_percentile=99, dendrite_percentile=98):
    """
    Generate skeleton from ZYXarray and remove skeleton points connected to soma regions.
    
    Args:
        ZYXarray (numpy.ndarray): Original 3D image data
        threshold_area_um2 (float): Area threshold in square microns for filtering soma
        x_um, y_um, z_um (float): Pixel sizes in microns
        soma_percentile (float): Percentile for soma detection (default 99)
        dendrite_percentile (float): Percentile for dendrite skeletonization (default 98)
    
    Returns:
        tuple: (skeleton_3d, skeleton_points_3d, dilated_soma_mask)
    """
    from scipy.ndimage import median_filter, binary_opening, binary_closing, label, binary_dilation
    from scipy.ndimage import generate_binary_structure
    from scipy.spatial import cKDTree
    
    # Denoise
    denoised_ZYXarray = median_filter(ZYXarray, size=(1, 5, 5))
    
    # Binarize for soma detection (higher percentile)
    binary_ZYXarray_soma = denoised_ZYXarray > np.percentile(denoised_ZYXarray, soma_percentile)
    
    # Binarize for dendrite detection (lower percentile)
    binary_ZYXarray_dendrite = denoised_ZYXarray > np.percentile(denoised_ZYXarray, dendrite_percentile)
    
    # Morphological operations for soma detection
    structure = generate_binary_structure(rank=3, connectivity=1)
    opened_soma = binary_opening(binary_ZYXarray_soma, structure=structure)
    closed_soma = binary_closing(opened_soma, structure=structure)
    
    # Label connected components for soma detection
    labeled_array, num_features = label(closed_soma, structure=structure)
    
    # Define voxel size threshold
    min_voxel_count = threshold_area_um2/(x_um*y_um)
    
    # Create soma mask (large components)
    soma_mask = np.zeros_like(closed_soma, dtype=bool)
    for i in range(1, num_features + 1):
        component_voxels = (labeled_array == i)
        if component_voxels.sum() >= min_voxel_count:
            soma_mask[component_voxels] = True
    
    # Dilate soma mask by 5 pixels to include nearby skeleton points
    dilated_soma_mask = binary_dilation(soma_mask, structure=np.ones((3,21,21)))
    
    # Generate skeleton from dendrite binary image
    skeleton_3d, skeleton_points_3d = skeletonize_with_branch_filtering(
        binary_ZYXarray_dendrite, 
        min_branch_length=10, 
        min_component_size=10, 
        connection_radius=2
    )
    
    # Remove skeleton points that overlap with dilated soma mask
    skeleton_points_coords = np.array(np.where(skeleton_3d)).T
    if len(skeleton_points_coords) > 0:
        # Check which skeleton points are in soma regions
        soma_overlap = dilated_soma_mask[tuple(skeleton_points_coords.T)]
        valid_skeleton_points = skeleton_points_coords[~soma_overlap]
        
        # Create new skeleton without soma-connected points
        skeleton_3d_clean = np.zeros_like(skeleton_3d)
        if len(valid_skeleton_points) > 0:
            skeleton_3d_clean[tuple(valid_skeleton_points.T)] = True
        
        return skeleton_3d_clean, valid_skeleton_points, dilated_soma_mask
    
    return skeleton_3d, skeleton_points_3d, dilated_soma_mask



def get_and_save_candidate_pos(flim_path, **kwargs):
    params = {
        "threshold_area_um2": 10,
        "spacing_um": 30,
        "ch_specified": False,
        "ch1or2": 1,
        "dendrite_percentile": 98,
        "soma_percentile": 99,
        "max_pos_cand_num": 6
    }
    params.update(kwargs)
    
    iminfo = FileReader()
    iminfo.read_imageFile(flim_path, True)
    eachpos_export_path = os.path.join(Path(flim_path).parent,
                                       Path(flim_path).stem)    
    os.makedirs(eachpos_export_path, exist_ok=True)

    pos_pix_csv_path = os.path.join(eachpos_export_path, "assigned_pixel_pos.csv")
    pos_um_csv_path = os.path.join(eachpos_export_path, "assigned_relative_um_pos.csv")

    ZYXarray = np.array(iminfo.image).sum(axis=tuple([1,2,5]))
    x_um, y_um, z_um = get_xyz_pixel_um(iminfo)
    # Generate skeleton from soma-filtered image
    skeleton_3d, _, _ = generate_skeleton_from_filtered_image(
        ZYXarray, params["threshold_area_um2"], x_um, y_um, z_um, 
        soma_percentile=params["soma_percentile"], dendrite_percentile=params["dendrite_percentile"]
    )
    
    surface_z = find_surface_1d(ZYXarray)
    skeleton_points = define_skeleton_points(skeleton_3d, params["spacing_um"], x_um, y_um, z_um)
    
    sorted_skeleton_points = skeleton_points[skeleton_points[:, 0].argsort()[::-1]] = skeleton_points[skeleton_points[:, 0].argsort()[::-1]]
    
    sorted_skeleton_points = sorted_skeleton_points[:params["max_pos_cand_num"]]
    
    plot_skeleton_with_points(skeleton_3d, sorted_skeleton_points, change_aspect_ratio=True,
                                saveplot=True, savepath=os.path.join(eachpos_export_path, f"skeleton_with_points_skeleton.png"))
    plot_skeleton_with_points(ZYXarray, sorted_skeleton_points, change_aspect_ratio=True,
                                saveplot=True, savepath=os.path.join(eachpos_export_path, f"skeleton_with_points_ZYX.png"))
    
    save_pix_pos_from_click_list(sorted_skeleton_points, 
                                    csv_savepath=pos_pix_csv_path)
    ZYX_um_dict = get_abs_um_pos_from_center_3d(iminfo.statedict, sorted_skeleton_points)
    save_um_pos_from_click_list(ZYX_um_dict, 
                                csv_savepath=pos_um_csv_path)
    print(f"Saved candidate positions to \n{pos_pix_csv_path} \n{pos_um_csv_path}")


# %%
if __name__ == "__main__":
    # flim_path = r"G:\ImagingData\Tetsuya\20250612\test_lowmag\lowmag1_001_20250602_256_zoom2_GFP.flim"
    flim_path_list = [
        r"G:\ImagingData\Tetsuya\20250619\auto1\lowmag1_001.flim"
        ]
    threshold_area_um2 = 10
    spacing_um = 30
    ch_specified = False
    ch1or2 = 1
    percentile = 98
    
    for flim_path in flim_path_list:
        plt_savefolder = os.path.join(Path(flim_path).parent,
                                       'plot_skeleton_3d')
        os.makedirs(plt_savefolder, exist_ok=True)
        min_dendrite_length_um = 25
        iminfo = FileReader()
        iminfo.read_imageFile(flim_path, True)
        # ZYXarray = np.array(iminfo.image)[:,:,ch1or2 - 1,:,:,:].sum(axis=tuple([1,4]))
        ZYXarray = np.array(iminfo.image).sum(axis=tuple([1,2,5]))
        x_um, y_um, z_um = get_xyz_pixel_um(iminfo)
        # Generate skeleton from soma-filtered image
        skeleton_3d, skeleton_points_3d, dilated_soma_mask = generate_skeleton_from_filtered_image(
            ZYXarray, threshold_area_um2, x_um, y_um, z_um, 
            soma_percentile=99, dendrite_percentile=percentile
        )
        
        skeleton_3d_dilated = binary_dilation(skeleton_3d, structure=np.ones((1,3,3)))
        skeleton_points = define_skeleton_points(skeleton_3d, spacing_um, x_um, y_um, z_um)
        
        print(f"Number of skeleton points: {len(skeleton_points)}")
        print(f"Spacing: {spacing_um} um")
        
        plot_skeleton_3d(skeleton_3d_dilated, x_um, y_um, z_um, saveplot=False)
        plot_skeleton_with_points(skeleton_3d, skeleton_points, change_aspect_ratio=True,
                                  saveplot=True, savepath=os.path.join(plt_savefolder, f"{os.path.basename(flim_path)}_skeleton_with_points_skeleton.png"))
        plot_skeleton_with_points(ZYXarray, skeleton_points, change_aspect_ratio=True,
                                  saveplot=True, savepath=os.path.join(plt_savefolder, f"{os.path.basename(flim_path)}_skeleton_with_points_ZYX.png"))
        
        break
#         import numpy as np
#         from scipy.ndimage import binary_opening, generate_binary_structure, binary_closing, label


#         structure = generate_binary_structure(rank=3, connectivity=1)  # 6-connectivity
#         opened = binary_opening(binary_ZYXarray, structure=structure)
#         closed = binary_closing(opened, structure=structure)
#         labeled_array, num_features = label(closed, structure=structure)
#         # Define voxel size threshold
#         min_voxel_count = threshold_area_um2/(x_um*y_um)
#         # Create an output array initialized to False
#         filtered_image = np.zeros_like(closed, dtype=bool)
#         # Keep only components larger than threshold
#         for i in range(1, num_features + 1):
#             component_voxels = (labeled_array == i)
#             if component_voxels.sum() >= min_voxel_count:
#                 filtered_image[component_voxels] = True
#         # To visualize one slice
#         import matplotlib.pyplot as plt
#         plt.subplot(1,4,1)
#         plt.imshow(ZYXarray.max(axis=0), cmap='gray')
#         plt.title("Original")
#         plt.axis('off')
#         plt.subplot(1,4,2)
#         plt.imshow(binary_ZYXarray.max(axis=0), cmap='gray')
#         plt.title("Binary")
#         plt.axis('off')
#         plt.subplot(1,4,3)
#         plt.imshow(closed.max(axis=0), cmap='gray')
#         plt.title("Open/Close")
#         plt.axis('off')
#         plt.subplot(1,4,4)
#         plt.imshow(filtered_image.max(axis=0), cmap='gray')
#         plt.title("Filtered")
#         plt.axis('off')
#         plt.show()


# # %%

# %%
