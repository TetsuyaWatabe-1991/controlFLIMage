# %%
import os
import sys
sys.path.append('..\\')
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import tifffile
from matplotlib import pyplot as plt
import seaborn as sns
from FLIMageFileReader2 import FileReader
from simple_dialog import ask_yes_no_gui, ask_save_path_gui, ask_open_path_gui
from skimage.segmentation import find_boundaries
from simple_dialog import ask_save_folder_gui
from scipy.ndimage import median_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator

def plt_zpro_with_roi(
    mask_containing_df_single,
    roi_types, color_dict, vmax, vmin,
    highmag_side_length_um,ch_1or2=1):

    flim_filepath = mask_containing_df_single["file_path"]
    corrected_uncaging_z = mask_containing_df_single["corrected_uncaging_z"]

    iminfo = FileReader()
    iminfo.read_imageFile(flim_filepath, True)
    six_dim = np.array(iminfo.image)

    z_from = int(mask_containing_df_single["z_from"])
    z_to = int(mask_containing_df_single["z_to"])

    z_projection = six_dim[z_from:z_to,0,ch_1or2 - 1,:,:,:].sum(axis=-1).max(axis=0)

    plt.imshow(z_projection, cmap="gray",interpolation="none",
                vmax=vmax, vmin=vmin,
                extent=(0, highmag_side_length_um, highmag_side_length_um, 0)
                # extent=(0, z_projection.shape[1], z_projection.shape[0], 0)
                )
    linewidth = 0.5
    for each_roi_type in roi_types:
        temp_mask = mask_containing_df_single[f"{each_roi_type}_shifted_mask"]
        boundaries = find_boundaries(temp_mask, mode='thick')
        height, width = temp_mask.shape
        for y in range(height):
            for x in range(width):
                if boundaries[y, x]:
                    plot_pos_x = x * highmag_side_length_um/z_projection.shape[1]
                    plot_pos_y = y * highmag_side_length_um/z_projection.shape[0]
                    increment_x = highmag_side_length_um/z_projection.shape[1]
                    increment_y = highmag_side_length_um/z_projection.shape[0]
                    if y == 0 or temp_mask[y-1, x] != temp_mask[y, x]:
                        plt.plot([plot_pos_x, plot_pos_x+increment_x], [plot_pos_y, plot_pos_y], color=color_dict[each_roi_type], linewidth=linewidth)
                    if y == height-1 or temp_mask[y+1, x] != temp_mask[y, x]:
                        plt.plot([plot_pos_x, plot_pos_x+increment_x], [plot_pos_y+increment_y, plot_pos_y+increment_y], color=color_dict[each_roi_type], linewidth=linewidth)
                    if x == 0 or temp_mask[y, x-1] != temp_mask[y, x]:
                        plt.plot([plot_pos_x, plot_pos_x], [plot_pos_y, plot_pos_y+increment_y], color=color_dict[each_roi_type], linewidth=linewidth)
                    if x == width-1 or temp_mask[y, x+1] != temp_mask[y, x]:
                        plt.plot([plot_pos_x+increment_x, plot_pos_x+increment_x], [plot_pos_y, plot_pos_y+increment_y], color=color_dict[each_roi_type], linewidth=linewidth)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Y (um)")
    plt.xlabel("X (um)")







def plt_GCaMP_F_F0_with_roi(
    fig_dim,
    subplot_num_list,
    each_label_unc_df,
    F_F0_vmin=1, F_F0_vmax=10, cmap="inferno"):

    flim_filepath = each_label_unc_df["file_path"].iloc[0]
    center_x = each_label_unc_df["center_x"].iloc[0]
    center_y = each_label_unc_df["center_y"].iloc[0]

    uncaging_iminfo = FileReader()
    uncaging_iminfo.read_imageFile(flim_filepath, True) 
    imagearray = np.array(uncaging_iminfo.image)

    if imagearray.shape[0] in [4, 33, 34]:
        print("Summed GCaMP")
        GCpre = imagearray[0,0,0,:,:,:].sum(axis=-1)
        GCunc = imagearray[3,0,0,:,:,:].sum(axis=-1)
    elif imagearray.shape[0] in [32]:
        print("no summation")
        GCpre = imagearray[8*0 + 1 : 8*1, 0,0,:,:,:].sum(axis=-1).sum(axis=0)
        GCunc = imagearray[8*3 + 1 : 8*4, 0,0,:,:,:].sum(axis=-1).sum(axis=0)

    # Apply median filter
    GC_pre_med = median_filter(GCpre, size=3)
    GC_unc_med = median_filter(GCunc, size=3)
    vmax_for_raw = GC_unc_med.max()
    
    # Calculate F/F0
    GCF_F0 = (GC_unc_med/GC_pre_med)
    GCF_F0[GC_pre_med == 0] = 0

    marker_size = 10    

    plt.subplot(fig_dim[0], fig_dim[1], subplot_num_list[0])
    im = plt.imshow(GCpre, cmap='gray',interpolation="none",
                vmax=vmax_for_raw, vmin=0)
    plt.plot(center_x, center_y, 'c+', markersize=marker_size)
    plt.axis("off")
    plt.title("Pre")

    plt.subplot(fig_dim[0], fig_dim[1], subplot_num_list[1])
    im = plt.imshow(GCunc, cmap='gray',interpolation="none",
                vmax=vmax_for_raw, vmin=0)
    plt.plot(center_x, center_y, 'c+', markersize=marker_size)
    plt.axis("off")
    plt.title("Uncaging")


    plt.subplot(fig_dim[0], fig_dim[1], subplot_num_list[2])
    im = plt.imshow(GCF_F0, cmap=cmap,interpolation="none",
                vmin=F_F0_vmin, vmax=F_F0_vmax)
    plt.plot(center_x, center_y, 'c+', markersize=marker_size)
    plt.axis("off")
    plt.title("GCaMP F/F0")

    # inset_axes で横長のカラーバー用のAxesを追加
    cax = inset_axes(plt.gca(),
                    width="40%",  
                    height="100%",  
                    loc='upper center',
                    bbox_to_anchor=(0.0, -0.03, 1.0, 0.05), # (x0, y0, width, height)
                    bbox_transform=plt.gca().transAxes,
                    borderpad=1)

    # カラーバーの描画
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    #no ticks
    cbar.ax.set_xticks([])
    cbar.ax.set_yticks([])
    #plot vmin and vmax at the each side of the cbar
    cbar.ax.text(F_F0_vmin, 0.5, f"{F_F0_vmin}", ha='right', va='center', fontsize=10, color="black")
    cbar.ax.text(F_F0_vmax, 0.5, f"{F_F0_vmax}", ha='left', va='center', fontsize=10, color="black")


    # img = mpimg.imread(r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250701\auto1\plot\hori_1to10.png")
    # imagebox = inset_axes(
    #     plt.gca(),
    #     width="100%", height="100%",
    #     loc='lower center',
    #     bbox_to_anchor=(0.0, 1.0, 1.0, 0.2), # (x0, y0, width, height)
    #     bbox_transform=plt.gca().transAxes,
    #     borderpad=0
    # )
    # imagebox.imshow(img)
    # imagebox.axis('off')  # 軸を非表示




# one_of_filepath_dict = {
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250531\4lines_2_auto\lowmag1__highmag_1_002.flim":"4_lines",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250531\2lines_1\lowmag1__highmag_1_002.flim":"2_lines",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250601\2lines3_auto\lowmag1__highmag_1_002.flim":"2_lines",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250601\4lines_3_auto\lowmag1__highmag_1_002.flim":"4_lines",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250602\lowmag1__highmag_1_002.flim":"2_lines",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250602\4lines_neuron4\lowmag1__highmag_1_002.flim":"4_lines",
# }

# one_of_filepath_dict = {
#     r"C:\Users\WatabeT\Desktop\20250701\auto1\lowmag2__highmag_3_089.flim":"GCaMP",
# }

ch_1or2 = 2
one_of_filepath_dict = {
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250714\auto1\lowmag1__highmag_7_047.flim":"GCaMP",
}


# combined_df_reject_bad_data_pkl_path = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250531\combined\combined_df_reject_bad_data.pkl"
# combined_df_reject_bad_data_pkl_path = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250531\combined\combined_df_2.pkl"

# LTP_point_df_pkl_path = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250531\combined\LTP_point_df.pkl"

# LTP_point_df_pkl_path = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250527\2ndlowmag\LTP_point_df.pkl"
# combined_df_reject_bad_data_pkl_path = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250527\2ndlowmag\combined_df_reject_bad_data.pkl"
# combined_df_reject_bad_data_pkl_path = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250527\2ndlowmag\combined_df_2.pkl"

# combined_df_reject_bad_data_pkl_path =r"C:\Users\WatabeT\Desktop\20250701\auto1\combined_df_reject_bad_data.pkl"
# LTP_point_df_pkl_path =r"C:\Users\WatabeT\Desktop\20250701\auto1\LTP_point_df.pkl"
# combined_df_reject_bad_data_pkl_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\combined_df_reject_bad_data.pkl"
# LTP_point_df_pkl_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\LTP_point_df.pkl"


combined_df_reject_bad_data_pkl_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250714\auto1\combined_df_reject_bad_data.pkl"
combined_df_pkl_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250714\auto1\combined_df.pkl"
LTP_point_df_pkl_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250714\auto1\LTP_point_df.pkl"


savefolder = ask_save_folder_gui()



combined_df_reject_bad_data_df = pd.read_pickle(combined_df_reject_bad_data_pkl_path)
combined_df = pd.read_pickle(combined_df_pkl_path)

LTP_point_df = pd.read_pickle(LTP_point_df_pkl_path)

dt_formatter = "%Y-%m-%dT%H:%M:%S.%f"


if ask_yes_no_gui("Do you have lowmag_df?"):
    lowmag_df_path = ask_open_path_gui(filetypes=[("Pickle files","*.pkl")])
    lowmag_df = pd.read_pickle(lowmag_df_path)
else:
    lowmag_df = pd.DataFrame()
    save_images = ask_yes_no_gui("Do you want to save lowmag images?")
    for file_path, value in one_of_filepath_dict.items():
        print(file_path)
        lowmag_file_list = [
            f for f in glob.glob(
                os.path.join(
                    os.path.dirname(file_path),
                    "*[0-9][0-9][0-9].flim"
                )
            )
            if "highmag" not in os.path.basename(f).lower()  # .lower() は任意
        ]

        tiff_array_savefolder = os.path.join(os.path.dirname(file_path), "lowmag_tiff_array")
        z_projection_savefolder = os.path.join(os.path.dirname(file_path), "lowmag_tiff_z_projection")
        y_projection_savefolder = os.path.join(os.path.dirname(file_path), "lowmag_tiff_y_projection")
        for each_savefolder in [tiff_array_savefolder, z_projection_savefolder, y_projection_savefolder]:
            os.makedirs(each_savefolder, exist_ok=True)

        for one_of_file in lowmag_file_list:
            # print(one_of_file)
            iminfo = FileReader()
            # try:
            try:
                iminfo.read_imageFile(one_of_file, save_images)
            except Exception as e:
                print("error, could not read file")
                print(one_of_file)
                print(e)
                continue


            dt_str = iminfo.acqTime[0]
            acq_dt = datetime.strptime(dt_str, dt_formatter)
            statedict = iminfo.statedict.copy()

            tiff_array_savepath = os.path.join(tiff_array_savefolder, os.path.basename(one_of_file).replace(".flim", ".tif"))
            z_projection_savepath = os.path.join(z_projection_savefolder, os.path.basename(one_of_file).replace(".flim", ".tif"))
            y_projection_savepath = os.path.join(y_projection_savefolder, os.path.basename(one_of_file).replace(".flim", ".tif"))

            if save_images:
                six_dim = np.array(iminfo.image)
                each_tiff_array = six_dim[:,0, ch_1or2 - 1,:,:,:].sum(axis=-1)
                # print(each_tiff_array.shape)
                tifffile.imwrite(tiff_array_savepath, each_tiff_array)
                tifffile.imwrite(z_projection_savepath, each_tiff_array.max(axis=0))
                tifffile.imwrite(y_projection_savepath, each_tiff_array.max(axis=1))


            each_df = pd.DataFrame({
                "highmag_one_of_file_path": [file_path],
                "lowmag_file_path": [one_of_file],
                "acq_dt": [acq_dt],
                "lowmag_tiff_array_savepath": [tiff_array_savepath],
                "lowmag_tiff_z_projection_savepath": [z_projection_savepath],
                "lowmag_tiff_y_projection_savepath": [y_projection_savepath],
                "statedict": [statedict],
                })

            lowmag_df = pd.concat([lowmag_df, each_df], ignore_index=True)

    lowmag_df_path = ask_save_path_gui()
    if lowmag_df_path[:-4] != ".pkl":
        lowmag_df_path = lowmag_df_path + ".pkl"
    lowmag_df.to_pickle(lowmag_df_path)

# %%

for index, row in combined_df_reject_bad_data_df.iterrows():
    each_highmag_path = combined_df_reject_bad_data_df.at[index, "file_path"]
    highmag_acq_dt = combined_df_reject_bad_data_df.at[index, "dt"]
    str_index_highmag = each_highmag_path.rfind("_highmag_")
    lowmag_header = each_highmag_path[:str_index_highmag]

    possible_lowmag_df = lowmag_df[lowmag_df["lowmag_file_path"].str.contains(lowmag_header, regex=False)]

    #get filepath which acq_dt is just before highmag_acq_dt
    possible_lowmag_df = possible_lowmag_df[possible_lowmag_df["acq_dt"] < highmag_acq_dt]
    earliest_lowmag_file_path = possible_lowmag_df.iloc[0]["lowmag_file_path"]
    if len(possible_lowmag_df) == 0:
        print("no lowmag file found")
        continue

    combined_df_reject_bad_data_df.at[index, "lowmag_file_path"] = earliest_lowmag_file_path
    combined_df_reject_bad_data_df.at[index, "lowmag_tiff_array_savepath"] = possible_lowmag_df.iloc[0]["lowmag_tiff_array_savepath"]
    combined_df_reject_bad_data_df.at[index, "lowmag_tiff_z_projection_savepath"] = possible_lowmag_df.iloc[0]["lowmag_tiff_z_projection_savepath"]
    combined_df_reject_bad_data_df.at[index, "lowmag_tiff_y_projection_savepath"] = possible_lowmag_df.iloc[0]["lowmag_tiff_y_projection_savepath"]

    each_roi_png_savefolder = earliest_lowmag_file_path[:-5]
    each_roi_png_num_zfill3 = each_highmag_path[str_index_highmag+9:each_highmag_path.rfind("_")].zfill(3)
    lowmag_assigned_roi_png = os.path.join(each_roi_png_savefolder, f"{each_roi_png_num_zfill3}.png")
    assert os.path.exists(lowmag_assigned_roi_png)
    combined_df_reject_bad_data_df.at[index, "lowmag_assigned_roi_png"] = lowmag_assigned_roi_png
    combined_df_reject_bad_data_df.at[index, "each_roi_png_savefolder"] = each_roi_png_savefolder



# sns.lineplot(x='binned_min', y="norm_intensity", W
#     data=combined_df_reject_bad_data_df,
#     errorbar="se", legend=True,
#     linewidth=5, color="red")


# six_dim = np.array(iminfo.image)
# %%
time_min_threshold = 25

highmag_zoom = 15
highmag_pixel = 128
highmag_z_um = 1
highmag_z_len = 11
highmag_nAveFrame = 3

unique_id = 0


for each_label in combined_df_reject_bad_data_df["label"].unique():

# each_label = combined_df_reject_bad_data_df["label"].unique()[0]
    each_label_df = combined_df_reject_bad_data_df[(combined_df_reject_bad_data_df["label"] == each_label) &
                                                    (combined_df_reject_bad_data_df["phase"] != "unc")]

    if len(each_label_df) == 0:
        continue
    rejected_bool = each_label_df["reject"].iloc[0]

    group = each_label_df["group"].iloc[0]

    each_lowmag_df = lowmag_df[lowmag_df["lowmag_file_path"] == each_label_df["lowmag_file_path"].iloc[0]]
    assert len(each_lowmag_df) == 1

    lowmag_zoom = each_lowmag_df["statedict"].iloc[0]["State.Acq.zoom"]
    lowmag_side_length_um_zoom1 = each_lowmag_df["statedict"].iloc[0]["State.Acq.FOV_default"][0]
    lowmag_z_um = each_lowmag_df["statedict"].iloc[0]["State.Acq.sliceStep"]
    lowmag_z_len = each_lowmag_df["statedict"].iloc[0]["State.Acq.nSlices"]
    lowmag_nAveFrame = each_lowmag_df["statedict"].iloc[0]["State.Acq.nAveFrame"]

    rel_um_df = pd.read_csv(each_label_df['each_roi_png_savefolder'].iloc[0] + "/assigned_relative_um_pos.csv")

    group_num = group[
        group.rfind("_highmag_") + len("_highmag_"):
        group.rfind("_")
        ]


    lowmag_side_length_um = lowmag_side_length_um_zoom1 / lowmag_zoom
    lowmag_z_len_um = lowmag_z_len * lowmag_z_um

    highmag_z_len_um = highmag_z_len * highmag_z_um
    highmag_side_length_um = lowmag_side_length_um_zoom1/highmag_zoom

    each_group_rel_um_df = rel_um_df[rel_um_df["pos_id"] == int(group_num)]

    x_left_um = np.clip(each_group_rel_um_df["x_um"].iloc[0]+lowmag_side_length_um/2 - highmag_side_length_um/2, 0, lowmag_side_length_um)
    y_lower_um = np.clip(each_group_rel_um_df["y_um"].iloc[0]+lowmag_side_length_um/2 - highmag_side_length_um/2, 0, lowmag_side_length_um)
    z_lower_um = np.clip(lowmag_z_len_um/2 - each_group_rel_um_df["z_um"].iloc[0] - highmag_z_len_um/2, 0, lowmag_z_len_um)

    x_right_um = np.clip(each_group_rel_um_df["x_um"].iloc[0]+lowmag_side_length_um/2 + highmag_side_length_um, 0, lowmag_side_length_um)
    y_upper_um = np.clip(each_group_rel_um_df["y_um"].iloc[0]+lowmag_side_length_um/2 + highmag_side_length_um, 0, lowmag_side_length_um)
    z_upper_um = np.clip(lowmag_z_len_um/2 - each_group_rel_um_df["z_um"].iloc[0] + highmag_z_len_um, 0, lowmag_z_len_um)

    before_align_highmag_zproj_txy_array = tifffile.imread(each_label_df['before_align_save_path'].iloc[0])
    before_unc_nth = each_label_df[each_label_df['phase'] == 'pre'].iloc[-1]["nth_omit_induction"]  - each_label_df[each_label_df['phase'] == 'pre'].iloc[0]["nth_omit_induction"]
    before_uncaging_zproj = before_align_highmag_zproj_txy_array[before_unc_nth, :, :]
    corrected_unc_x = each_label_df[each_label_df['phase'] == 'pre'].iloc[-1]["corrected_uncaging_x"]
    corrected_unc_y = each_label_df[each_label_df['phase'] == 'pre'].iloc[-1]["corrected_uncaging_y"]

    small_x_from = each_label_df[each_label_df['phase'] == 'pre'].iloc[-1]["small_x_from"]
    small_y_from = each_label_df[each_label_df['phase'] == 'pre'].iloc[-1]["small_y_from"]
    unc_x_for_plot = corrected_unc_x - small_x_from
    unc_y_for_plot = corrected_unc_y - small_y_from
    unc_x_for_plot_um = unc_x_for_plot * highmag_side_length_um/highmag_pixel
    unc_y_for_plot_um = unc_y_for_plot * highmag_side_length_um/highmag_pixel


    # center_x
    vmax = before_uncaging_zproj.max()/3
    vmin = before_uncaging_zproj.min()
    
    lowmag_vmax = vmax * (lowmag_nAveFrame/highmag_nAveFrame) 
    lowmag_vmin = vmin * (lowmag_nAveFrame/highmag_nAveFrame) 


    roi_types = ["Spine", "DendriticShaft", "Background"]
    color_dict = {"Spine": "red", "DendriticShaft": "blue", "Background": "green"}

   ########################################################
    each_size = 4
    fig_dim = [2,6]
    ylim = [-0.4,2.1]
    xlim = [-11,36]
    z_plus_minus = 1

    plt.figure(figsize=(each_size*fig_dim[1], each_size*fig_dim[0]))  # Increased height from 10 to 20
    if rejected_bool:
        plt.suptitle(each_label + "\n rejected  " + each_label_df.comment.iloc[0])
    else:
        plt.suptitle(each_label)

    ########################################################

    plt.subplot(fig_dim[0], fig_dim[1], 1)
    plt.title("Low mag  Z projection")
    zpro_array = tifffile.imread(each_label_df['lowmag_tiff_z_projection_savepath'].iloc[0])
    plt.imshow(zpro_array, cmap="gray", vmin = lowmag_vmin, vmax = lowmag_vmax,
            extent=(0, lowmag_side_length_um, lowmag_side_length_um, 0))
    plt.plot([x_left_um, x_right_um, x_right_um, x_left_um, x_left_um],
            [y_lower_um, y_lower_um, y_upper_um, y_upper_um, y_lower_um], "r-", markersize=10)

    plt.ylabel("Y (um)")
    plt.xlabel("X (um)")



    ########################################################

    plt.subplot(fig_dim[0], fig_dim[1], 7)
    plt.title("Low mag  Y projection")
    ypro_array = tifffile.imread(each_label_df['lowmag_tiff_y_projection_savepath'].iloc[0])
    # flipped = np.flip(ypro_array, axis=0)
    plt.imshow(ypro_array, cmap="gray", vmin = lowmag_vmin, vmax = lowmag_vmax,
                origin="lower",
                extent=(0, lowmag_side_length_um, ypro_array.shape[0] *lowmag_z_um,0),
                )
    plt.plot([x_left_um, x_right_um, x_right_um, x_left_um, x_left_um],
            [z_lower_um, z_lower_um, z_upper_um, z_upper_um, z_lower_um], "r-", markersize=10)
    plt.ylabel("Z (um)")
    plt.xlabel("X (um)")


    ########################################################

    plt.subplot(fig_dim[0], fig_dim[1], 2)
    roi_png = plt.imread(each_label_df['lowmag_assigned_roi_png'].iloc[0])

    plt.imshow(roi_png,
                extent=(0, lowmag_side_length_um, lowmag_side_length_um, 0))
    plt.title("ROI")
    plt.ylabel("Y (um)")
    plt.xlabel("X (um)")



    ########################################################
    plt.subplot(fig_dim[0], fig_dim[1], 8)
    plt.title("Uncaging")
    trimmed_high_x_um = highmag_side_length_um/highmag_pixel * before_uncaging_zproj.shape[1]
    trimmed_high_y_um = highmag_side_length_um/highmag_pixel * before_uncaging_zproj.shape[0]
    plt.imshow(before_uncaging_zproj, cmap="gray", vmin = vmin*2, vmax = vmax*2,
                extent=(0, trimmed_high_x_um, trimmed_high_y_um, 0))
    plt.plot(unc_x_for_plot_um, unc_y_for_plot_um, "c+", markersize=10)
    plt.ylabel("Y (um)")
    plt.xlabel("X (um)")

    ########################################################
    plt.subplot(fig_dim[0], fig_dim[1], 3)

    mask_containing_df_single = each_label_df[each_label_df['phase'] == 'pre'].iloc[-1]

    plt_zpro_with_roi(
        mask_containing_df_single,
        roi_types, color_dict, vmax, vmin,
        highmag_side_length_um,ch_1or2)
    plt.title("Pre " + str(mask_containing_df_single["relative_time_min"]) + "min" + f"  Z: {int(mask_containing_df_single['z_from']+1)} to {int(mask_containing_df_single['z_to'])}")

    ########################################################
    plt.subplot(fig_dim[0], fig_dim[1], 4)
    mask_containing_df_single = each_label_df[each_label_df['phase'] == 'post'].iloc[0]

    plt_zpro_with_roi(
        mask_containing_df_single,
        roi_types, color_dict, vmax, vmin,
        highmag_side_length_um,ch_1or2)
    plt.title("Post " + str(mask_containing_df_single["relative_time_min"]) + "min" + f"  Z: {int(mask_containing_df_single['z_from']+1)} to {int(mask_containing_df_single['z_to'])}")

    ########################################################
    plt.subplot(fig_dim[0], fig_dim[1], 5)
    
    mask_containing_df_single_candidate = each_label_df[each_label_df['relative_time_min'] > time_min_threshold]
    if len(mask_containing_df_single_candidate) == 0:
        mask_containing_df_single_candidate = each_label_df.loc[each_label_df['relative_time_min'].idxmax():,:]
    mask_containing_df_single = mask_containing_df_single_candidate.iloc[0]
    data_used_index_for_plot = mask_containing_df_single_candidate.index[0]

    plt_zpro_with_roi(
        mask_containing_df_single,
        roi_types, color_dict, vmax, vmin,
        highmag_side_length_um,ch_1or2)
    plt.title("Post " + str(mask_containing_df_single["relative_time_min"]) + "min" + f"  Z: {int(mask_containing_df_single['z_from']+1)} to {int(mask_containing_df_single['z_to'])}")


    ########################################################
    plt.subplot(fig_dim[0], fig_dim[1], 6)
    plt.plot(
        xlim, [0,0],
        color="gray", linestyle="--", zorder= -10
        )
    plt.text(0, (ylim[1] - ylim[0])*0.90 + ylim[0],
    "uncaging", color="k", fontsize=10, va="bottom", ha = "left")

    plt.plot(each_label_df['relative_time_min'],
            each_label_df['norm_intensity'],
            color="k",marker="o",linestyle="-", zorder=0
            )
    
    # data_used_index = each_label_df[each_label_df['relative_time_min'] > time_min_threshold].index[0]
    plt.scatter(each_label_df.at[data_used_index_for_plot, "relative_time_min"],
            each_label_df.at[data_used_index_for_plot, "norm_intensity"],
            color="m",marker="o",s=100, zorder=10
            )
    text_pos_x = np.clip(each_label_df.at[data_used_index_for_plot, "relative_time_min"], xlim[0], xlim[1])
    text_pos_y = np.clip(each_label_df.at[data_used_index_for_plot, "norm_intensity"], ylim[0], ylim[1])
    plt.text(text_pos_x, text_pos_y,
            str(round(each_label_df.at[data_used_index_for_plot, "norm_intensity"], 2)),
            color = "m",  fontsize=20, zorder=100,
            va="bottom",ha="left"
            )

    plt.plot([0,0],
        [ (ylim[1] - ylim[0])*0.90 + ylim[0],
        (ylim[1] - ylim[0])*0.85 + ylim[0]],
        color="r"
        )

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.ylabel("normalized intensity")
    plt.xlabel("Time (min)")


    ########################################################
    

    each_label_unc_df = combined_df[(combined_df["label"] == each_label)&
                                                    (combined_df["phase"] == "unc")]
    F_F0_vmin = 1
    F_F0_vmax = 10
    cmap = "inferno"
    subplot_num_list = [9,10,11]
    plt_GCaMP_F_F0_with_roi(
        fig_dim,
        subplot_num_list,
        each_label_unc_df,
        F_F0_vmin, F_F0_vmax, cmap)
      

    ########################################################
    plt.subplot(fig_dim[0], fig_dim[1], 12)

    GCaMP_Spine_subBG = each_label_unc_df['GCaMP_Spine_subBG'].values[0]
    GCaMP_DendriticShaft_subBG = each_label_unc_df['GCaMP_DendriticShaft_subBG'].values[0]

    GCaMP_Spine_F_F0 = GCaMP_Spine_subBG/(GCaMP_Spine_subBG[:2].mean())
    GCaMP_DendriticShaft_F_F0 = GCaMP_DendriticShaft_subBG/(GCaMP_DendriticShaft_subBG[:2].mean())


    flim_filepath = each_label_unc_df["file_path"].iloc[0]
    unc_iminfo = FileReader()
    unc_iminfo.read_imageFile(flim_filepath, False) 
    GCaMP_Spine_subBG

    #time data '2025-07-03T16:52:27.509' does not match format '%Y-%m-%d %H:%M:%S'
    acq_time_list = [datetime.strptime(each_time, "%Y-%m-%dT%H:%M:%S.%f") for each_time in unc_iminfo.acqTime]
    delta_sec_list = [(each_time - acq_time_list[0]).total_seconds() for each_time in acq_time_list]

    plt.plot(delta_sec_list,
            GCaMP_Spine_F_F0,
            color="r",marker="o",linestyle="-", zorder=0,
            label="Spine"
            )
    plt.plot(delta_sec_list,
            GCaMP_DendriticShaft_F_F0,
            color="b",marker="o",linestyle="-", zorder=0,
            label="DendriticShaft"
            )
    ylim_base = [0,21]
    plt.ylim(ylim_base)
 
    plt.ylabel("GCaMP F/F0")
    plt.xlabel("Time (sec)")
    plt.legend()


    ########################################################

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    unique_id+=1

    wo_date_savefolder = os.path.join(savefolder, "wo_date")
    with_date_savefolder = os.path.join(savefolder, "with_date")
    os.makedirs(wo_date_savefolder, exist_ok=True)
    os.makedirs(with_date_savefolder, exist_ok=True)

    savepath_wo_date = os.path.join(wo_date_savefolder, f"{os.path.basename(each_label)}_{str(unique_id)}.png")
    savepath_with_date = os.path.join(with_date_savefolder, f"{each_label_df.dt.iloc[0].strftime("%Y%m%d_%H%M%S")}_{os.path.basename(each_label)}_{str(unique_id)}.png")
    plt.savefig(savepath_wo_date, dpi=150, bbox_inches="tight")
    plt.savefig(savepath_with_date, dpi=150, bbox_inches="tight")

    plt.show()



print("done")






# %%
reject_norm_intensity_threshold = 3
LTP_point_df_cut_too_large = LTP_point_df[LTP_point_df["norm_intensity"] < reject_norm_intensity_threshold]

for ind, row in LTP_point_df_cut_too_large.iterrows():
    each_label = row["label"]
    for key, value in one_of_filepath_dict.items():
        if os.path.dirname(each_label) in key:
            LTP_point_df_cut_too_large.at[ind, "two_groups"] = value
            break


for each_group in LTP_point_df_cut_too_large["two_groups"].unique():
    each_group_df = LTP_point_df_cut_too_large[LTP_point_df_cut_too_large["two_groups"] == each_group]
    mean_norm_intensity = each_group_df["norm_intensity"].mean()
    std_norm_intensity = each_group_df["norm_intensity"].std()
    print(each_group)
    print("number of data: ", len(each_group_df))
    print(mean_norm_intensity)
    print(std_norm_intensity)
    print("--------------------------------")

sns.swarmplot(x="two_groups", y="norm_intensity",
    data=LTP_point_df_cut_too_large)
#delete upper and right axes
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#delta spine volume
plt.ylabel("$\Delta$ spine volume")
plt.xlabel("")






# %%
