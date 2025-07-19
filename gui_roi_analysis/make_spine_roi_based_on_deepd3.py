import os
import numpy as np
from tifffile import imread, imsave
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from skimage.morphology import binary_opening, binary_closing, disk, erosion, dilation

def create_roi_from_spine_and_shaft(spine_path, shaft_path, 
                                    spine_threshold=0.15, 
                                    shaft_threshold=0.10,
                                    watershed_threshold=0.5,
                                    plot_them=True,
                                    save_them=True,
                                    save_path=None,
                                    roi_3d_TF = False,
                                    shaft_z_thickness = 3):
    spine_img = imread(spine_path)
    shaft_img = imread(shaft_path)
    print("spine_img.shape", spine_img.shape)
    print("shaft_img.shape", shaft_img.shape)
    if roi_3d_TF:
        spine_img = spine_img.max(axis=0)
        shaft_z = int(shaft_img.shape[0]/2)
        if shaft_z_thickness ==1:
            shaft_img = shaft_img[shaft_z,:,:]
        else:
            shaft_z_from = max(0, shaft_z - shaft_z_thickness//2)
            shaft_z_to = min(shaft_img.shape[0], shaft_z + shaft_z_thickness//2)
            shaft_img = shaft_img[shaft_z_from:shaft_z_to,:,:].max(axis=0)

        
        # z_axis = np.argmin(spine_img.shape)
        # spine_img = spine_img.max(axis=z_axis)
        # shaft_z = int(shaft_img.shape[z_axis]/2)

        # indices = [slice(None)] * shaft_img.ndim
        # indices[z_axis] = shaft_z
        # shaft_img = shaft_img[tuple(indices)]
        
    print("spine_img.shape", spine_img.shape)
    print("shaft_img.shape", shaft_img.shape)
    # Threshold processing -> Create mask
    spine_mask = spine_img > spine_threshold
    shaft_mask = shaft_img > shaft_threshold

    # ImageJ-style preprocessing: Morphological operations to remove noise
    # Remove small noise
    spine_mask = binary_opening(spine_mask, disk(1))
    # Fill holes
    spine_mask = binary_closing(spine_mask, disk(2))
    
    shaft_mask = binary_opening(shaft_mask, disk(1))
    shaft_mask = binary_closing(shaft_mask, disk(2))

    spine_minus_shaft_mask = spine_mask & ~shaft_mask

    # ImageJ-style watershed implementation
    # 1. Distance transform (ImageJ-style: more detailed settings)
    distance = ndi.distance_transform_edt(spine_minus_shaft_mask)
    
    # 2. Create ImageJ-style markers (simpler approach)
    # Normalize the distance transform result to 8-bit (ImageJ style)
    distance_array = np.array(distance)
    distance_norm = ((distance_array - distance_array.min()) / (distance_array.max() - distance_array.min()) * 255).astype(np.uint8)
    
    # 3. Detect local maxima (method similar to ImageJ's watershed)
    # Detect local maxima with a larger filter size
    local_max = ndi.maximum_filter(distance_norm, size=5) == distance_norm
    local_max = local_max & (distance_norm > watershed_threshold * 255)
    
    # 4. Label markers
    markers = label(local_max)
    
    # 5. Execute watershed (ImageJ style)
    # ImageJ's watershed usually runs without markers to separate objects
    # First, execute watershed without markers (ImageJ's default behavior)
    spine_watershed = watershed(-distance_array, mask=spine_minus_shaft_mask, 
                               connectivity=2, compactness=0)
    
    # If markers are present, try the marked watershed
    markers_array = np.array(markers)
    if np.sum(markers_array) > 0:
        # Marked watershed (if more precise control is needed)
        spine_watershed_marked = watershed(-distance_array, markers, mask=spine_mask, 
                                          connectivity=2, compactness=0)
        # Use the marked watershed if it gives better results
        if np.max(spine_watershed_marked) > np.max(spine_watershed):
            spine_watershed = spine_watershed_marked
    
    # Strengthen separation between objects: 1 pixel apart
    # Process each label individually
    unique_labels = np.unique(spine_watershed)
    if len(unique_labels) > 1:  # If there are labels other than background
        separated_watershed = np.zeros_like(spine_watershed)
        
        # Check adjacent relationships
        eroded_labels = []  # Record of eroded labels
        
        for label_id in unique_labels[1:]:  # Process background (0)
            # Create mask for each label
            label_mask = (spine_watershed == label_id)
            
            # Check if this label has adjacent labels
            # Check if it expands to touch other labels
            dilated_mask = dilation(label_mask, disk(1))
            other_labels = np.unique(spine_watershed[dilated_mask])
            other_labels = other_labels[other_labels != label_id]  # Exclude self
            other_labels = other_labels[other_labels != 0]  # Exclude background
            
            if len(other_labels) > 0:
                # If there are adjacent objects: Erode by 1 pixel
                eroded = erosion(label_mask, disk(1))
                separated_watershed[eroded] = label_id
                eroded_labels.append(label_id)
                print(f"Label {label_id}: Adjacent objects found → Erode processing executed")
            else:
                # If there are no adjacent objects: keep as is
                separated_watershed[label_mask] = label_id
                print(f"Label {label_id}: No adjacent objects → keep as is")
        
        spine_watershed = separated_watershed
        
        if len(eroded_labels) > 0:
            print(f"Number of eroded objects: {len(eroded_labels)}")
        else:
            print("No adjacent objects found")
    

    # Labeling and ROI extraction
    labeled = label(spine_watershed)
    regions = regionprops(labeled)

    # Output example of ROI (Bounding box)
    rois = []
    for region in regions:
        if region.area >= 10:  # Remove too small noise
            rois.append(region.bbox)  # (min_row, min_col, max_row, max_col)

    if plot_them:
        #plot spine_mask, shaft_mask, spine_watershed
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(spine_mask, cmap='gray')
        axes[0, 0].set_title("Spine Mask")
        axes[0, 0].axis('off')
        axes[0, 1].imshow(shaft_mask, cmap='gray')
        axes[0, 1].set_title("Shaft Mask")
        axes[0, 2].imshow(distance_norm, cmap='gray')
        axes[0, 2].set_title("Distance Transform (8-bit)")
        axes[0, 2].axis('off')
        axes[1, 0].imshow(local_max, cmap='gray')
        axes[1, 0].set_title("Local Maxima")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(spine_minus_shaft_mask, cmap='gray')
        axes[1, 1].set_title("Spine Minus Shaft Mask")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(spine_watershed, cmap='nipy_spectral')
        axes[1, 2].set_title("Spine Watershed (Erode if Adjacent)")
        axes[1, 2].axis('off')
        plt.tight_layout()
        if save_them:
            plt.savefig(save_path)
        plt.show()
        

        print(f"Detected Spine Number: {len(np.unique(spine_watershed)) - 1}")  # Background (0) is excluded

    return rois, labeled, regions, spine_watershed


if __name__ == "__main__":
    # example
    # spine_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\lowmag2__highmag_2\lowmag2__highmag_2_19_S_spine.tif"
    # shaft_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\lowmag2__highmag_2\lowmag2__highmag_2_19_S_shaft.tif"
    # shaft_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\tif\deepd3\lowmag1__highmag_7__3_S_shaft.tif"
    # spine_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\tif\deepd3\lowmag1__highmag_7__3_S_spine.tif"

    shaft_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\tif\deepd3\lowmag1__highmag_3__2_S_shaft.tif"
    spine_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\tif\deepd3\lowmag1__highmag_3__2_S_spine.tif"
    original_image_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\tif\lowmag1__highmag_3__2.0_after_align.tif"


    import glob
    shaft_list = glob.glob(r"C:\Users\WatabeT\Desktop\20250701\auto1\tif\deepd3\*S_shaft.tif")

    for each_shaft_path in shaft_list:
        each_spine_path = each_shaft_path.replace("shaft", "spine")
        each_original_image_stem = os.path.basename(each_shaft_path.replace("_S_shaft.tif", ".0_after_align.tif"))
        each_original_image_path = os.path.join(r"C:\Users\WatabeT\Desktop\20250701\auto1\tif", each_original_image_stem)
        each_savepath = os.path.join(r"C:\Users\WatabeT\Desktop\20250701\auto1\tif\deepd3_overlay", 
                                    each_original_image_stem + "_overlay.png")
        assert os.path.exists(each_original_image_path), f"original image not found: {each_original_image_path}"
        assert os.path.exists(each_shaft_path), f"shaft image not found: {each_shaft_path}"
        assert os.path.exists(each_spine_path), f"spine image not found: {each_spine_path}"
        
        rois, labeled, regions, spine_watershed = create_roi_from_spine_and_shaft(
            each_spine_path, each_shaft_path, plot_them=True, save_them=True, save_path=each_savepath[:-11] + "_watershed.png")
        labeled_tif_path = os.path.join(r"C:\Users\WatabeT\Desktop\20250701\auto1\tif\label_tif", 
                                        each_original_image_stem + "_labeled.tif")
        imsave(labeled_tif_path, labeled)

        #read original image and max projection
        original_image = imread(each_original_image_path)
        max_projection = original_image.max(axis=0)


        fig, axes = plt.subplots(1, 3, figsize=(15, 5))


        axes[0].imshow(max_projection, cmap='gray')
        axes[0].set_title("Max Projection")
        axes[0].axis('off')


        axes[1].imshow(spine_watershed, cmap='nipy_spectral')
        axes[1].set_title("Watershed Result")
        axes[1].axis('off')


        axes[2].imshow(max_projection, cmap='gray')
        for each_region in regions:
            each_coords = each_region.coords
            each_coords_x = each_coords[:, 1]
            each_coords_y = each_coords[:, 0]

            axes[2].scatter(each_coords_x, each_coords_y, c='r', s=1)

        axes[2].set_title("Spine ROIs")
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(each_savepath)
        plt.show()

        print(f"Number of Detected ROIs: {len(rois)}")