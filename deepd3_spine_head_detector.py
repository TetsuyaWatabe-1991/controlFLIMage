import os
import math
import sys
sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from deepd3.core.analysis import Stack, ROI3D_Creator
from matplotlib.colors import ListedColormap
from skimage.measure import  regionprops_table
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
from FLIMageFileReader2 import FileReader
from FLIMageAlignment import get_xyz_pixel_um
from scipy.ndimage import grey_closing, median_filter
from skimage.exposure import equalize_adapthist
from skimage.morphology import disk, white_tophat


def ensure_directory(dirpath: str) -> str:
    """Create dirpath if missing; return path when it already exists."""
    if os.path.isdir(dirpath):
        return dirpath
    if os.path.isfile(dirpath):
        alt = dirpath + "_save"
        print(f"  Note: {dirpath!r} is a file; using {alt!r}")
        os.makedirs(alt, exist_ok=True)
        return alt
    try:
        os.makedirs(dirpath, exist_ok=True)
    except FileExistsError:
        if os.path.isdir(dirpath):
            return dirpath
        raise
    return dirpath


def resolve_savefolder(flim_path: str) -> str:
    """
    Resolve output folder for a .flim file (multi_spine / mushroom convention).

    Prefer legacy path flim_path[:-9] (strip _NNN.flim). On some Windows/SMB
  shares that name cannot be created when the .flim sits in the same directory
    (e.g. AP5_pos6_256_4x_001.flim blocks folder AP5_pos6_256_4x). Then use
    the full .flim stem directory (…/AP5_pos6_256_4x_001).
    """
    legacy = flim_path[:-9]
    if os.path.isdir(legacy):
        return legacy
    try:
        os.makedirs(legacy, exist_ok=True)
        if os.path.isdir(legacy):
            return legacy
    except OSError:
        pass
    stem = os.path.splitext(flim_path)[0]
    ensure_directory(stem)
    if stem != legacy:
        print(
            f"  Note: cannot use {legacy!r}; save folder is {stem!r} "
            f"(.flim sibling name conflict on this volume)"
        )
    return stem


def local_z_mip_stack(zyx, radius=2):
    """
    Per-Z max over +/-radius neighbors (2*radius+1 planes); edge-padded at stack ends.
    """
    zyx = np.asarray(zyx, dtype=np.float32)
    radius = int(radius)
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    if radius == 0:
        return zyx.copy()
    padded = np.pad(zyx, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    win = 2 * radius + 1
    out = np.empty_like(zyx)
    for z in range(zyx.shape[0]):
        out[z] = np.max(padded[z : z + win], axis=0)
    return out


def preprocess_stack_for_thin_branches(zyx, mode="tophat_clahe"):
    """
    Boost thin bright processes before DeepD3 inference.
    mode: 'none' | 'median' | 'tophat_clahe'
    """
    zyx = np.asarray(zyx, dtype=np.float32)
    if mode == "none":
        return zyx
    den = median_filter(zyx, size=(1, 3, 3))
    if mode == "median":
        return den
    out = den.copy()
    radius = 2
    selem = disk(radius)
    for z in range(out.shape[0]):
        plane = out[z]
        if plane.max() <= plane.min():
            continue
        pmin, pmax = float(plane.min()), float(plane.max())
        plane_n = (plane - pmin) / (pmax - pmin)
        enhanced = white_tophat(plane_n, selem)
        combined = np.maximum(plane_n, enhanced)
        out[z] = equalize_adapthist(combined.astype(np.float64), clip_limit=0.03)
    return out.astype(np.float32)


def fuse_dendrite_with_image_mask(
    dendrite_pred,
    raw_zyx,
    image_percentile=92.0,
    fusion_weight=0.5,
    closing_iterations=1,
):
    """
    Merge low-threshold image mask into dendrite prediction so thin branches
    visible in raw data appear in shaft map (low-res / 256_4x stacks).
    """
    dendrite_pred = np.asarray(dendrite_pred, dtype=np.float32)
    raw_zyx = np.asarray(raw_zyx, dtype=np.float32)
    den = median_filter(raw_zyx, size=(1, 3, 3))
    thresh = np.percentile(den, image_percentile)
    image_mask = (den >= thresh).astype(np.float32)
    fused = np.maximum(dendrite_pred, fusion_weight * image_mask)
    if closing_iterations > 0:
        for _ in range(closing_iterations):
            fused = grey_closing(fused, size=(1, 3, 3))
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


class SpinePosDeepD3():
    def __init__(self, 
                 trainingdata_path = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DeepD3_32F.h5",
                 **kwargs):
        self.trainingdata_path = trainingdata_path
        self.params = {
                       "roi_mode": "thresholded",
                       "roi_areaThreshold" : 0.5,
                       "roi_peakThreshold" : 0.9,
                       "roi_seedDelta" : 0.1,
                       "roi_distanceToSeed" : 10,
                       "min_roi_size" : 5,
                       "max_roi_size" : 1000,
                       "min_planes" : 1,
                       "max_dist_spine_dend_um": 3,
                       "dendrite_skeleton_threshold": 0.9,
                       "enhance_thin_branches": False,
                       "stack_preprocess": "none",
                       "image_fusion_percentile": 92.0,
                       "image_fusion_weight": 0.5,
                       "dendrite_closing_iterations": 1,
                       "use_local_z_mip": False,
                       "local_z_mip_radius": 2,
                       }
        
        for eachkey in kwargs:
            self.params[eachkey] = kwargs[eachkey]
    
    def calculate_orientation_from_coordinates(self, coordinates):
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        
        # Fit a first-degree polynomial (line) to the coordinates
        coefficients = np.polyfit(x, y, 1)
        
        # Calculate the angle of the line
        orientation = np.arctan(coefficients[0])
        
        # Map the angle to the range -pi/2 to pi/2
        if orientation < -np.pi / 2:
            orientation += np.pi
        elif orientation > np.pi / 2:
            orientation -= np.pi
        
        return orientation
    
    
    def calculate_orientation(self,point1, point2):
        vector = np.array(point2) - np.array(point1)
        orientation = np.arctan2(vector[1], vector[0])
        if orientation <= -np.pi / 2:
            orientation += np.pi
        elif orientation > np.pi / 2:
            orientation -= np.pi
        return orientation
    
    
    def glasbey_colors(self):
        glasbey_colors = [
            (0,0,0),
            (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
            (1.0, 0.4980392156862745, 0.054901960784313725),
            (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
            (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
            (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
            (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
            (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
            (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
            (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
            (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
            (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
            (1.0, 0.4980392156862745, 0.054901960784313725),
            (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
            (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
            (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
            (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
            (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
            (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
            (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
            (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)
        ]
        glasbey_cmap = ListedColormap(glasbey_colors)
        return glasbey_cmap
    
    def plot_uncaging_pos(self, S, r, prop_dict, cand_spines, 
                          skeleton, result_dict, savefolder,
                          savefilename, show_interactive=True):
        vmax = S.stack.max()
        if S.stack.shape[0] == 1:
            fig, axs = plt.subplots(1, 4, figsize=(4, 1))
            axs = axs.reshape(1, -1)
        else:
            fig, axs = plt.subplots(S.stack.shape[0], 4, figsize=(4, S.stack.shape[0]))
        for each_z in range(S.stack.shape[0]):
            axs[each_z, 0].imshow(S.stack[each_z,:,:], 
                                vmin=0, vmax=vmax,
                                cmap='gray')
            axs[each_z, 1].imshow(S.stack[each_z,:,:], 
                                vmin=0, vmax=vmax,
                                cmap='gray')
            axs[each_z, 2].imshow(S.stack[each_z,:,:], 
                                vmin=0, vmax=vmax,
                                cmap='gray')
            axs[each_z, 3].imshow(skeleton, 
                                vmin=0, vmax=1,
                                cmap='gray')
            for n in range(4):
                axs[each_z,n].axis("off")
                axs[each_z,n].tick_params(axis='both', which='major', labelsize=5)
        
            for eachlabel in prop_dict:
                if prop_dict[eachlabel]["z"] == each_z:    
                    if eachlabel in cand_spines.index:                
                        axs[each_z, 1].plot(cand_spines.loc[eachlabel,'x'],
                                            cand_spines.loc[eachlabel,'y'],
                                            "r.", ms = 1)
                        
            for each_detected in result_dict:
                if result_dict[each_detected]["z_pix"] == each_z:
                    axs[each_z, 2].scatter(result_dict[each_detected]["x_pix"], 
                                           result_dict[each_detected]["y_pix"],
                                           c='y', s=0.02, marker = "+")
        
        savefolder = ensure_directory(savefolder)
        savepath = os.path.join(savefolder, f"{savefilename}_roi.png")
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
        if show_interactive:
            plt.show()
        plt.close(fig)
        print("save as ", savepath)
        
        fig, axs = plt.subplots(1, 6, figsize = (6,1))
        axs[0].imshow(np.amax(S.stack,axis=0), 
                    cmap="gray")
        axs[0].axis("off")
        
        axs[1].imshow(np.amax(S.stack,axis=0), 
                    cmap="gray")
        axs[1].axis("off")
        
        axs[2].imshow(np.amax(r.roi_map,axis=0), 
                    cmap = self.glasbey_colors())
        axs[2].axis("off")
        
        axs[3].imshow(np.amax(r.roi_map,axis=0), 
                    cmap = self.glasbey_colors())
        axs[3].axis("off")

        axs[4].imshow(np.amax(S.stack,axis=0), 
                    cmap="gray")
        axs[4].imshow(np.amax(r.roi_map,axis=0), 
                    cmap = self.glasbey_colors(),
                    alpha = (np.amax(r.roi_map,axis=0)>0).astype(np.float32)) 
        axs[4].axis("off")

        axs[5].imshow(np.amax(S.stack,axis=0), 
            cmap="gray")
        axs[5].axis("off")

        
        for eachlabel in prop_dict:
            x = prop_dict[eachlabel]["x"]
            y = prop_dict[eachlabel]["y"]
            num_pixels = prop_dict[eachlabel]["num_pixels"]
            
            # nearest_x = prop_dict[prop_df.loc[eachlabel, 'nearest_point_label']]['x']
            # nearest_y = prop_dict[prop_df.loc[eachlabel, 'nearest_point_label']]['y']
            #axs[1].plot([x, nearest_x], [y, nearest_y],'w:', lw = 0.2)
            
            #axs[1].text((x + nearest_x)/2,
            #            (y + nearest_y)/2,
            #            str(round(prop_df.loc[eachlabel]["distance_to_nearest"],1)),
            #            fontsize=2, color='yellow',va = 'center',ha = 'center')
            
            axs[2].text(x,y,str(num_pixels),
                                fontsize=2, color='white')
            
            intensity = prop_dict[eachlabel]["intensity"]        
            axs[3].text(x,y,str(intensity),
                    fontsize=2, color='white')
            
            if eachlabel in cand_spines.index:
                axs[ 1].plot(x,y,
                             "r.",ms = 1)
                print(eachlabel)
            else:
                axs[ 1].plot(x,y,
                             "c.",ms = 1)
                print(eachlabel)
                
        for each_detected in result_dict:
            axs[5].scatter(result_dict[each_detected]["x_pix"], 
                           result_dict[each_detected]["y_pix"],
                           c='y', s=0.02, marker = "+")
        #fig.suptitle("area : " + str(roi_areaThreshold)
        #        +"  peak : "+ str(roi_peakThreshold), size = 7) 
        # savepath = os.path.join(savefolder,f"result_area{roi_areaThreshold}_peak{roi_peakThreshold}_mip.png")
        savepath = os.path.join(savefolder, f"{savefilename}_mip.png")
        plt.savefig(savepath, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print("save as ", savepath)

    def save_detection_outputs(self, S, r, prop_dict, cand_spines, skeleton,
                               result_dict, save_folder, save_filename_stem,
                               save_overview_pngs=True,
                               save_prediction_stacks=False, skeleton_3d=False,
                               show_interactive=False):
        """Save DeepD3 overview PNGs and optional shaft/spine prediction TIFFs."""
        save_folder = ensure_directory(save_folder)
        if save_prediction_stacks:
            spine_path = os.path.join(
                save_folder, f"{save_filename_stem}_S_spine.tif"
            )
            shaft_path = os.path.join(
                save_folder, f"{save_filename_stem}_S_shaft.tif"
            )
            tifffile.imwrite(spine_path, S.prediction[..., 1])
            tifffile.imwrite(shaft_path, S.prediction[..., 0])
            print("save as ", spine_path)
            print("save as ", shaft_path)

        if save_overview_pngs:
            if skeleton.ndim == 3:
                skeleton_for_plot = skeleton.max(axis=0)
            else:
                skeleton_for_plot = skeleton
            self.plot_uncaging_pos(
                S, r, prop_dict, cand_spines, skeleton_for_plot, result_dict,
                save_folder, save_filename_stem,
                show_interactive=show_interactive,
            )

    def return_uncaging_pos_based_on_roi_sum(self,
                                             flim_path = "",
                                             direct_ZYXarray_use = False,
                                             direct_ZYXarray = None,
                                             xy_pixel_um = None,
                                             z_pixel_um = None,
                                             max_distance = True,
                                             plot_them = True,
                                             specify_flim_ch = False,
                                             ch_1or2 = 2,
                                             upper_lim_spine_pixel_percentile = 60,
                                             lower_lim_spine_pixel_percentile = 30,
                                             upper_lim_spine_intensity_percentile = 70,
                                             lower_lim_spine_intensity_percentile = 10,
                                             ignore_first_n_plane=1,
                                             ignore_last_n_plane=1,
                                             ignore_edge_percentile = 5,
                                             savefolder = "",
                                             define_save_folder=False,
                                             save_folder="",
                                             save_filename_stem="",
                                             save_prediction_stacks=False,
                                             skeleton_3d=False,
                                             return_outputs_context=False,
                                             skeleton_3d_detection=None,
                                             ):
        if skeleton_3d_detection is None:
            skeleton_3d_detection = skeleton_3d
        
        if direct_ZYXarray_use == False:
            assert os.path.exists(flim_path), f"flim_path not found: {flim_path}"
            iminfo = FileReader()
            iminfo.read_imageFile(flim_path, True)
            if specify_flim_ch:
                assert ch_1or2 in [1,2], f"ch_1or2 is not 1 or 2: {ch_1or2}"
                ZYXarray = np.array(iminfo.image)[:,:,ch_1or2-1,:,:,:].sum(axis=tuple([1,4]))
            else:
                ZYXarray = np.array(iminfo.image).sum(axis=tuple([1,2,5]))
            x_um, _, z_um = get_xyz_pixel_um(iminfo)
            self.params["xy_pixel_um"] = x_um
            self.params["z_pixel_um"] = z_um
            ext = flim_path.split(".")[-1]
            if define_save_folder:
                assert save_folder, "save_folder must be set when define_save_folder=True"
                assert save_filename_stem, (
                    "save_filename_stem must be set when define_save_folder=True"
                )
                output_folder = save_folder
                output_stem = save_filename_stem
            elif os.path.exists(savefolder):
                output_folder = savefolder
                output_stem = os.path.basename(savefolder)
            else:
                output_folder = flim_path[:-len(ext) - 1]
                output_stem = os.path.basename(output_folder)
        else:
            ZYXarray = direct_ZYXarray
            assert ZYXarray.shape
            assert type(xy_pixel_um) == float, f"xy_pixel_um is not float: {xy_pixel_um}"
            assert type(z_pixel_um) == float, f"z_pixel_um is not float: {z_pixel_um}"
            assert os.path.exists(savefolder), f"savefolder not found: {savefolder}"
            self.params["xy_pixel_um"] = xy_pixel_um
            self.params["z_pixel_um"] = z_pixel_um
            output_folder = save_folder if save_folder else savefolder
            output_stem = save_filename_stem or os.path.basename(output_folder)

        zyx_raw = np.asarray(ZYXarray, dtype=np.float32)
        if self.params.get("use_local_z_mip", False):
            z_radius = int(self.params.get("local_z_mip_radius", 2))
            zyx_for_deepd3 = local_z_mip_stack(zyx_raw, radius=z_radius)
            print(
                f"  local Z-MIP input: radius={z_radius} "
                f"({2 * z_radius + 1} planes per DeepD3 slice)"
            )
        else:
            zyx_for_deepd3 = zyx_raw

        stack_input = preprocess_stack_for_thin_branches(
            zyx_for_deepd3, mode=self.params.get("stack_preprocess", "none")
        )
        temp_output_path = BytesIO()
        tifffile.imwrite(temp_output_path, stack_input)
        
        print("Loading stack...")
        S = Stack(temp_output_path, 
                  dimensions=dict(xy=self.params["xy_pixel_um"], 
                                  z=self.params["z_pixel_um"])
                  )
        print("Training data path  ", self.trainingdata_path)
        print("Training data exists  ", os.path.exists(self.trainingdata_path))        
        S.predictWholeImage(self.trainingdata_path)
        dendrite_pred_raw = S.prediction[..., 0].copy()
        if self.params.get("enhance_thin_branches", False):
            dendrite_fused = fuse_dendrite_with_image_mask(
                dendrite_pred_raw,
                zyx_raw,
                image_percentile=self.params.get("image_fusion_percentile", 92.0),
                fusion_weight=self.params.get("image_fusion_weight", 0.5),
                closing_iterations=int(
                    self.params.get("dendrite_closing_iterations", 1)
                ),
            )
            S.prediction[..., 0] = dendrite_fused
            print(
                "  thin-branch enhance: fused dendrite frac>0.2="
                f"{np.mean(dendrite_fused > 0.2):.3f} "
                f"(raw pred {np.mean(dendrite_pred_raw > 0.2):.3f})"
            )
        else:
            dendrite_fused = dendrite_pred_raw
        roi_mode = self.params.get("roi_mode", "thresholded")
        print(f"Building 3D ROIs (mode={roi_mode})...")
        r = ROI3D_Creator(
            dendrite_prediction = S.prediction[..., 0],
            spine_prediction = S.prediction[..., 1],
            mode=roi_mode,
            areaThreshold = self.params["roi_areaThreshold"],
            peakThreshold = self.params["roi_peakThreshold"],
            seedDelta = self.params["roi_seedDelta"],
            distanceToSeed = self.params["roi_distanceToSeed"],
            dimensions=dict(xy = self.params["xy_pixel_um"],
                            z = self.params["z_pixel_um"])
            )
        r.create(self.params["min_roi_size"],
            self.params["max_roi_size"],
            self.params["min_planes"])
        prop_table = regionprops_table(r.roi_map,
                                       properties=['label',
                                                   "centroid",
                                                   'num_pixels',
                                                   'equivalent_diameter_area'])
        skel_thresh = self.params.get("dendrite_skeleton_threshold", 0.9)
        if skeleton_3d_detection:
            skeleton = skeletonize(S.prediction[..., 0] > skel_thresh)
        else:
            skeleton = skeletonize(S.prediction[..., 0].max(axis=0) > skel_thresh)
        skeleton_points = np.array(np.where(skeleton)).T
        prop_dict = {}
        intensity_list = []
        
        if (len(prop_table['label'])<2)+(len(skeleton_points)<2) :
            return {}
        
        for nthlabel in range(len(prop_table['label'])):
            z = prop_table['centroid-0'][nthlabel]
            y = prop_table['centroid-1'][nthlabel]
            x = prop_table['centroid-2'][nthlabel]
            num_pixels = prop_table['num_pixels'][nthlabel]
            intensity = ZYXarray[r.roi_map == prop_table['label'][nthlabel]].sum()
            name = str(prop_table['label'][nthlabel])
            prop_dict[name] = {"z":round(z),
                               "y":round(y),
                               "x":round(x),
                               "num_pixels": num_pixels,
                               "intensity": intensity,
                               "equivalent_diameter_area": prop_table['equivalent_diameter_area'][nthlabel]
                               }
            intensity_list.append(intensity)

        upper = np.percentile(prop_table['num_pixels'], upper_lim_spine_pixel_percentile)
        lower = np.percentile(prop_table['num_pixels'], lower_lim_spine_pixel_percentile)
        upper_intensity = np.percentile(intensity_list, upper_lim_spine_intensity_percentile)
        lower_intensity = np.percentile(intensity_list, lower_lim_spine_intensity_percentile)
        
        prop_df = pd.DataFrame.from_dict(prop_dict,orient = "index")
        
        points_matrix = prop_df[['x', 'y', 'z']].values
        points_matrix_um = points_matrix*[[self.params["xy_pixel_um"],
                                           self.params["xy_pixel_um"],
                                           self.params["z_pixel_um"]]]
        
        nearest_points = []
        nearest_points_label=[]
        for point in points_matrix_um:
            distances = cdist([point], points_matrix_um)
            nearest_index = np.argsort(distances)[0, 1]  # exclude the distance between itself
            nearest_points.append(points_matrix_um[nearest_index])
            nearest_points_label.append(prop_df.index[nearest_index])
        
        prop_df['nearest_point_coord'] = nearest_points
        prop_df['nearest_point_label'] = nearest_points_label
        prop_df['distance_to_nearest'] = [np.linalg.norm(a - b) for a, b in zip(points_matrix_um, nearest_points)]
        
        cand_spines = prop_df[(prop_df['num_pixels']<=upper)&
                              (prop_df['num_pixels']>=lower)&
                              (prop_df['intensity']<=upper_intensity)&
                              (prop_df['intensity']>=lower_intensity)]
        result_dict = {}
        
        
        min_x_coord = int(S.stack.shape[2]*ignore_edge_percentile/100)
        max_x_coord = int(S.stack.shape[2]*(1 - ignore_edge_percentile/100))
        min_y_coord = int(S.stack.shape[1]*ignore_edge_percentile/100)
        max_y_coord = int(S.stack.shape[1]*(1 - ignore_edge_percentile/100))
        
        for each_z in range(ignore_first_n_plane, 
                            S.stack.shape[0] - ignore_last_n_plane):
            for eachlabel in prop_dict:
                if prop_dict[eachlabel]["z"] == each_z:    
                    x = prop_dict[eachlabel]["x"]
                    y = prop_dict[eachlabel]["y"]
                    num_pixels = prop_dict[eachlabel]["num_pixels"]
                    if eachlabel in cand_spines.index:
                        if skeleton_3d_detection:
                            index = np.argmin(
                                cdist(np.array([[each_z, y, x]]), skeleton_points)
                            )
                            x_index, y_index = 2, 1
                        else:
                            index = np.argmin(cdist(np.array([[y, x]]), skeleton_points))
                            x_index, y_index = 1, 0
                        nearest_point = skeleton_points[index]
                        neighborhood_indices = np.argsort(
                            cdist([nearest_point], skeleton_points)
                        )
                        neighborhood_points = skeleton_points[
                            neighborhood_indices[0, :11]
                        ]
                        try:
                            orientation = self.calculate_orientation_from_coordinates(
                                neighborhood_points
                            )
                            x_moved = x - nearest_point[x_index]
                            y_moved = y - nearest_point[y_index]
                            x_rotated = x_moved*math.cos(orientation) - y_moved*math.sin(orientation)
                            if x_rotated<=0:
                                direction = 1
                            else:
                                direction = -1
                                
                            direction = direction*0.1
                            candi_x, candi_y = x, y
                            spine_bin = S.stack[each_z,:,:] > S.stack[each_z,y,x]*0.5
                            
                            for i in range(100):                            
                                if (int(candi_y) <= min_y_coord or int(candi_y) >= max_y_coord):
                                    break
                                if (int(candi_x) <= min_x_coord or int(candi_x) >= max_x_coord):
                                    break
                                
                                if max_distance:
                                    if (
                                        (self.params["xy_pixel_um"]
                                         * (nearest_point[x_index] - candi_x)) ** 2
                                        + (self.params["xy_pixel_um"]
                                           * (nearest_point[y_index] - candi_y)) ** 2
                                    ) > self.params["max_dist_spine_dend_um"] ** 2:
                                        break
                                    
                                if spine_bin[int(candi_y),int(candi_x)]>0:
                                    candi_x = candi_x - math.cos(orientation)*direction
                                    candi_y = candi_y + math.sin(orientation)*direction
                                else:
                                    result_dict[eachlabel] = {}
                                    result_dict[eachlabel]["x_pix"] = candi_x
                                    result_dict[eachlabel]["y_pix"] = candi_y
                                    result_dict[eachlabel]["z_pix"] = each_z
                                    result_dict[eachlabel]["neighborhood_points"] = neighborhood_points
                                    result_dict[eachlabel]["orientation"] = orientation
                                    result_dict[eachlabel]["direction"] = direction
                                    result_dict[eachlabel]["centroid_x_pix"] = prop_dict[eachlabel]["x"]
                                    result_dict[eachlabel]["centroid_y_pix"] = prop_dict[eachlabel]["y"]
                                    result_dict[eachlabel]['equivalent_diameter_area'] = prop_dict[eachlabel]['equivalent_diameter_area']
                                    break
                        except:
                            continue

        outputs_context = {
            "S": S,
            "r": r,
            "prop_dict": prop_dict,
            "cand_spines": cand_spines,
            "skeleton": skeleton,
            "output_folder": output_folder,
            "output_stem": output_stem,
            "dendrite_pred_raw": dendrite_pred_raw,
            "dendrite_pred_fused": dendrite_fused,
        }

        if plot_them or save_prediction_stacks:
            self.save_detection_outputs(
                S, r, prop_dict, cand_spines, skeleton, result_dict,
                output_folder, output_stem,
                save_overview_pngs=plot_them or save_prediction_stacks,
                save_prediction_stacks=save_prediction_stacks,
                skeleton_3d=skeleton_3d,
                show_interactive=plot_them,
            )
        if return_outputs_context:
            return result_dict, outputs_context
        return result_dict


if __name__ == "__main__":
    SpineAssign = SpinePosDeepD3()
    flim_path = r"G:\ImagingData\Tetsuya\20240530\AAVitB27_GFP7hrs_2ndN_dend2_015.flim"
    #flim_path = r"G:\ImagingData\Tetsuya\20240728\24well\highmagRFP50ms10p\tpem2\B1_00_1_2__highmag_2_007.flim"
    result_dict = SpineAssign.return_uncaging_pos_based_on_roi_sum(flim_path, plot_them = True)
    #for flim_path in glob.glob(r"G:\ImagingData\Tetsuya\20240530\*_015.flim"):
    #   result_dict = SpineAssign.return_uncaging_pos_based_on_roi_sum(flim_path, plot_them = True)
        