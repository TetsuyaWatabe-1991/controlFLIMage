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

class SpinePosDeepD3():
    def __init__(self, 
                 trainingdata_path = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DeepD3_32F.h5",
                 **kwargs):
        self.trainingdata_path = trainingdata_path
        self.params = {
                       "roi_areaThreshold" : 0.5,
                       "roi_peakThreshold" : 0.9,
                       "roi_seedDelta" : 0.1,
                       "roi_distanceToSeed" : 10,
                       "min_roi_size" : 5,
                       "max_roi_size" : 1000,
                       "min_planes" : 1,
                       "max_dist_spine_dend_um": 3}
        
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
                          savefilename):
        vmax = S.stack.max()
        fig, axs = plt.subplots(S.stack.shape[0], 4, figsize = (4,S.stack.shape[0]))       
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
        
        os.makedirs(savefolder, exist_ok=True)
        savepath = os.path.join(savefolder, f"{savefilename}_roi.png")
        plt.savefig(savepath,dpi=600, bbox_inches = 'tight')
        plt.show()
        # plt.close();plt.clf()
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
        plt.savefig(savepath,dpi=600, bbox_inches = 'tight')
        plt.close();plt.clf()
        
        plt.imshow(np.amax(S.stack,axis=0), 
                   cmap="gray")
        for each_detected in result_dict:
            plt.scatter(result_dict[each_detected]["x_pix"], 
                        result_dict[each_detected]["y_pix"],
                        c='y', s=10, marker = "+")
        plt.gca().axis("off")
        plt.show()

    def return_uncaging_pos_based_on_roi_sum(self,
                                             flim_path,
                                             max_distance = True,
                                             plot_them = True,
                                             upper_lim_spine_pixel_percentile = 60,
                                             lower_lim_spine_pixel_percentile = 30,
                                             upper_lim_spine_intensity_percentile = 70,
                                             lower_lim_spine_intensity_percentile = 10,
                                             ignore_first_n_plane=1,
                                             ignore_last_n_plane=1,
                                             ignore_edge_percentile = 5
                                             ):
        

        iminfo = FileReader()
        iminfo.read_imageFile(flim_path, True)
        ZYXarray = np.array(iminfo.image).sum(axis=tuple([1,2,5]))
        
        x_um, _, z_um = get_xyz_pixel_um(iminfo)
        self.params["xy_pixel_um"] = x_um
        self.params["z_pixel_um"] = z_um
        
        temp_output_path = BytesIO()
        tifffile.imwrite(temp_output_path, ZYXarray)
        
        ext = flim_path.split(".")[-1]
        print("Loading stack...")
        S = Stack(temp_output_path, 
                  dimensions=dict(xy=self.params["xy_pixel_um"], 
                                  z=self.params["z_pixel_um"])
                  )
        print("Training data path  ", self.trainingdata_path)
        print("Training data exists  ", os.path.exists(self.trainingdata_path))        
        S.predictWholeImage(self.trainingdata_path)        
        # plt.imshow(S.prediction[..., 0].max(axis=0)>=1,vmin=0,vmax=1)
        # plt.show()
        print("Building 3D ROIs...")
        r = ROI3D_Creator(
            dendrite_prediction = S.prediction[..., 0],
            spine_prediction = S.prediction[..., 1],
            mode = "thresholded",#'floodfill',
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
        skeleton = skeletonize(S.prediction[..., 0].max(axis=0) > 0.9)
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
                        index = np.argmin(cdist(np.array([[y,x]]), skeleton_points))
                        nearest_point = skeleton_points[index]
                        # # Sort by the order of the close distance
                        neighborhood_indices = np.argsort(cdist([nearest_point], skeleton_points))
                        # Get first 11 points coordinates
                        neighborhood_points = skeleton_points[neighborhood_indices[0,:11]]
                        try:
                            orientation = self.calculate_orientation_from_coordinates(neighborhood_points)
                            x_moved = x - nearest_point[1]
                            y_moved = y - nearest_point[0]
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
                                    if (((self.params["xy_pixel_um"]*(nearest_point[1] - candi_x))**2 
                                        + (self.params["xy_pixel_um"]*(nearest_point[0] - candi_y))**2)
                                            > self.params["max_dist_spine_dend_um"]**2):
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

        if plot_them:
            savefolder = flim_path[:-len(ext)-1]
            savefilename = os.path.basename(savefolder)
            self.plot_uncaging_pos(S, r, prop_dict, cand_spines, 
                                   skeleton, result_dict, savefolder, 
                                   savefilename)
        return result_dict          


if __name__ == "__main__":
    SpineAssign = SpinePosDeepD3()
    flim_path = r"G:\ImagingData\Tetsuya\20240530\AAVitB27_GFP7hrs_2ndN_dend2_015.flim"
    #flim_path = r"G:\ImagingData\Tetsuya\20240728\24well\highmagRFP50ms10p\tpem2\B1_00_1_2__highmag_2_007.flim"
    result_dict = SpineAssign.return_uncaging_pos_based_on_roi_sum(flim_path, plot_them = True)
    #for flim_path in glob.glob(r"G:\ImagingData\Tetsuya\20240530\*_015.flim"):
    #   result_dict = SpineAssign.return_uncaging_pos_based_on_roi_sum(flim_path, plot_them = True)
        