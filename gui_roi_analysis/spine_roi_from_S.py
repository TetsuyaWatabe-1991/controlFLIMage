# %%
import os
import pandas as pd
import datetime
from tifffile import imread
from matplotlib import pyplot as plt
import numpy as np
from make_spine_roi_based_on_deepd3 import create_roi_from_spine_and_shaft
from scipy.ndimage import label, find_objects
from scipy.ndimage import median_filter, binary_opening, binary_closing

SHIFT_DIRECTION = -1

def visualization_spine_roi_from_S(twoD_img, spine_watershed, regions, save_path = "", save_TF = False,
                                  unc_x = None, unc_y = None):
    #read original image and max projection

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(max_projection, cmap='gray')
    if unc_x is not None and unc_y is not None:
        axes[0].scatter(unc_x, unc_y, c='c', s=200, marker='x')
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
    if unc_x is not None and unc_y is not None:
        axes[2].scatter(unc_x, unc_y, c='c', s=200, marker='x')
    axes[2].set_title("Spine ROIs")
    axes[2].axis('off')
    plt.tight_layout()
    if save_TF:
        plt.savefig(save_path)
    plt.show()

# from utility.send_notification import send_slack_url_default
def find_closest_region(x, y, label_image, regions):
    # return the region that contains the point (x, y) or the closest region to the point (x, y)
    
    try:
        label_value = label_image[y, x] 

        if label_value > 0:
            for region in regions:
                if region.label == label_value:
                    return region
    except:
        print("Error: find_closest_region")
        print(f"label_image.shape: {label_image.shape}")
        print(f"y: {y}, x: {x}")
        print(f"regions: {regions}")
        pass

    min_dist = float('inf')
    closest_region = None
    for region in regions:
        cy, cx = region.centroid
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        if dist < min_dist:
            min_dist = dist
            closest_region = region

    return closest_region


def get_closest_points(binary_image, reference_point, top_n=50, avoid_top_n = 20):
    y_indices, x_indices = np.nonzero(binary_image)
    coords = np.stack((y_indices, x_indices), axis=1)
    distances = np.linalg.norm(coords - np.array(reference_point), axis=1)
    sorted_indices = np.argsort(distances)
    print('len(coords)', len(coords))
    print('len(sorted_indices)', len(sorted_indices))
    closest_coords = coords[sorted_indices[avoid_top_n:top_n]]
    print('len(closest_coords)', len(closest_coords))
    return closest_coords


def low_intensity_largest_mask(image, percentile=10):
    threshold = np.percentile(image, percentile)
    
    #apply median filter to image
    image = median_filter(image, size=5)
    low_mask = image <= threshold
    
    #apply open_close operation to low_mask
    low_mask = binary_opening(low_mask, structure=np.ones((3, 3)))
    low_mask = binary_closing(low_mask, structure=np.ones((3, 3)))

    low_mask[:image.shape[0]//20, :] = False
    low_mask[:, :image.shape[1]//20] = False
    low_mask[:, -image.shape[1]//20:] = False
    low_mask[-image.shape[0]//20:, :] = False

    labeled, num_features = label(low_mask)

    if num_features == 0:
        plt.imshow(image, cmap="gray")
        plt.show()
        plt.imshow(low_mask)
        plt.show()
        print("threshold", threshold)
        print("Make BG mask at the left top corner")
        mask = np.zeros_like(image, dtype=bool)
        mask[10:20,10:20] = True
        coords = np.argwhere(mask)
        return coords


    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    largest_label = np.argmax(sizes)
    if sizes[largest_label] == 0:
        raise ValueError("Largest region size is 0.")

    # slices = find_objects(labeled == largest_label)[0]
    # y1, y2 = slices[0].start, slices[0].stop
    # x1, x2 = slices[1].start, slices[1].stop

    mask = np.zeros_like(image, dtype=bool)
    mask[labeled == largest_label] = True
    coords = np.argwhere(mask)
    # mask[y1:y2, x1:x2] = True
    # coords = np.argwhere(mask)
    return coords


def define_roi_from_S(combined_df_path, save_path_suffix = "_with_roi_mask"):
    combined_df = pd.read_pickle(combined_df_path)
    roi_types = ["Spine", "DendriticShaft", "Background"]
    roi_param_suffixes = ["_roi_mask", "_roi_shape", "_roi_parameters"]
    for each_roi_type in roi_types:
        for each_roi_param in roi_param_suffixes:
            column_name = f"{each_roi_type}{each_roi_param}"
            if column_name not in combined_df.columns:
                combined_df[column_name] = pd.Series(dtype=object)
                print(f"{column_name} added")

    for each_filepath_without_number in combined_df["filepath_without_number"].unique():
        each_filepath_without_number_df = combined_df[combined_df["filepath_without_number"] == each_filepath_without_number]

        # ###############
        # ##for debug
        # if each_filepath_without_number != 'C:\\Users\\WatabeT\\Desktop\\20250701\\auto1\\lowmag4__highmag_2_':
        #     continue
        # ###############

        parent_dir = os.path.dirname(each_filepath_without_number) 
        deepd3_roi_folder = os.path.join(parent_dir, "deepd3_roi")   
        for each_nth_set_label in each_filepath_without_number_df["nth_set_label"].unique():
            if each_nth_set_label < 0:
                continue

            # ################
            # ###for debug    
            # if each_nth_set_label != 0:
            #     continue
            # ################

            try:
                each_nth_set_label_df_include_unc = each_filepath_without_number_df[each_filepath_without_number_df["nth_set_label"] == each_nth_set_label]
                unc_df = each_nth_set_label_df_include_unc[each_nth_set_label_df_include_unc["phase"] == "unc"]
                assert len(unc_df) == 1
                each_nth_set_label_df_without_unc = each_nth_set_label_df_include_unc[each_nth_set_label_df_include_unc["nth_omit_induction"] > -1]
                
                unc_x = unc_df.iloc[0].center_x - unc_df.iloc[0].small_x_from  + unc_df.iloc[0].small_shift_x*SHIFT_DIRECTION
                unc_y = unc_df.iloc[0].center_y - unc_df.iloc[0].small_y_from + unc_df.iloc[0].small_shift_y*SHIFT_DIRECTION
                

                tzyx_aligned_path_list = each_nth_set_label_df_include_unc.save_path_tzyx_aligned.unique()
                assert len(tzyx_aligned_path_list) == 1
                tzyx_aligned_path = tzyx_aligned_path_list[0]
                save_filename_stem = os.path.basename(tzyx_aligned_path).replace(".tif", "")
                print(each_filepath_without_number, each_nth_set_label, "\n" ,save_filename_stem)
            
                max_proj_spine_path = os.path.join(deepd3_roi_folder, save_filename_stem + "_max_proj_S_spine.tif")
                max_proj_shaft_path = os.path.join(deepd3_roi_folder, save_filename_stem + "_max_proj_S_shaft.tif")

                # min_proj_spine_path = os.path.join(deepd3_roi_folder, save_filename_stem + "_min_proj_S_spine.tif")
                # min_proj_shaft_path = os.path.join(deepd3_roi_folder, save_filename_stem + "_min_proj_S_shaft.tif")

                original_image = imread(tzyx_aligned_path)
                max_projection = original_image.max(axis=0).max(axis=0)
                save_folder = os.path.join(deepd3_roi_folder, "roi_visualization")
                os.makedirs(save_folder, exist_ok=True)

                shaft_img = imread(max_proj_shaft_path)
                z_from = max(0, shaft_img.shape[0]//2 -1)
                z_to = min(shaft_img.shape[0], shaft_img.shape[0]//2 + 2)
                shaft_2d = shaft_img[z_from:z_to, :, :].max(axis=0)

                rois, labeled, regions, spine_watershed = create_roi_from_spine_and_shaft(
                                    max_proj_spine_path, max_proj_shaft_path, plot_them=True, save_them=True, roi_3d_TF = True,
                                    save_path = os.path.join(save_folder, save_filename_stem + "max_and_max_watershed.png"))

                # visualization_spine_roi_from_S(twoD_img = max_projection, spine_watershed = spine_watershed, regions = regions, 
                #                             save_path = os.path.join(save_folder, save_filename_stem + "max_and_max_watershed_visualization.png"), 
                #                             save_TF = True, unc_x = unc_x, unc_y = unc_y)


                closest_region = find_closest_region(x = int(unc_x), y = int(unc_y), 
                                                    label_image = labeled, regions = regions)
                label_value = closest_region.label

                binary_spine_roi = labeled == label_value
                spine_coords = np.argwhere(binary_spine_roi)

                shaft_2d_bin = (shaft_2d > 0.5)&(~binary_spine_roi)
                shaft_coords = get_closest_points(shaft_2d_bin, [int(unc_y), int(unc_x)], 
                                                top_n=60, avoid_top_n=30)
                bg_coords = low_intensity_largest_mask(max_projection, percentile=10)

                ###############
                ##for debug visualization
                marker_size = 3
                plt.imshow(max_projection, cmap="gray")
                plt.plot(shaft_coords[:, 1], shaft_coords[:, 0], 'r.', markersize=marker_size, alpha=0.5)
                plt.plot(spine_coords[:, 1], spine_coords[:, 0], 'b.', markersize=marker_size, alpha=0.5)
                plt.plot(bg_coords[:, 1], bg_coords[:, 0], 'g.', markersize=marker_size, alpha=0.5)
                roi_visu_save_folder = os.path.join(save_folder, "three_rois_plot_on_img")
                os.makedirs(roi_visu_save_folder, exist_ok=True)
                plt.savefig(os.path.join(roi_visu_save_folder, 
                            save_filename_stem + "_plot_on_img.png"), dpi=300)
                plt.show()
                
                ###############

                for nth_omit_induction in each_nth_set_label_df_without_unc["nth_omit_induction"].unique():
                    each_nth_omit_induction_df = each_nth_set_label_df_without_unc[each_nth_set_label_df_without_unc["nth_omit_induction"] == nth_omit_induction]
                    
                    coords_dict = {"Spine": spine_coords, "DendriticShaft": shaft_coords, "Background": bg_coords}

                    for each_roi_type in coords_dict:
                        shifted_coords = coords_dict[each_roi_type].copy()
                        shifted_mask = np.zeros(shape = (max_projection.shape[0], max_projection.shape[1]), dtype=bool)
                        shifted_mask[shifted_coords[:, 0], shifted_coords[:, 1]] = True
                        idx = each_nth_omit_induction_df.index[0]
                        combined_df.at[idx, f"{each_roi_type}_roi_mask"] = shifted_mask.copy()
                        combined_df.at[idx, f"{each_roi_type}_roi_shape"] = "roi_from_S"
                        combined_df.at[idx, f"{each_roi_type}_roi_parameters"] = "roi_from_S"
                        combined_df.at[idx, f"{each_roi_type}_roi_area_pixels"] = np.sum(shifted_mask)
                        combined_df.at[idx, f"{each_roi_type}_quantified_datetime"] = datetime.datetime.now()
            except:
                print(f"Error: {each_filepath_without_number}, {each_nth_set_label}, {nth_omit_induction}")
                pass
                        

    savepath = combined_df_path.replace(".pkl", save_path_suffix + ".pkl")
    combined_df.to_pickle(savepath)


if __name__ == "__main__":
    combined_df_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250718\auto1\combined_df.pkl"
    define_roi_from_S(combined_df_path)

# %%