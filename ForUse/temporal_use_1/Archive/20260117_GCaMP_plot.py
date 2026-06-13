import sys
sys.path.append("../../")
from flimage_graph_func import plot_GCaMP_F_F0
import glob

file_list = [
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20260117\1203GCTom_2Ca1Mg_1_001.flim",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20260117\1203GCTom_2Ca1Mg_1_002.flim",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20260117\1203GCTom_2Ca1Mg_1_003.flim",
]


file_list = glob.glob(r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20260117\1203GCTom*.flim")


slope = 0.0729
intercept = 0.053

for each_file in file_list:
    plot_GCaMP_F_F0(each_file, slope = 0.0676, intercept = 0.146, 
                    from_Thorlab_to_coherent_factor = 1,
                    vmin = 1, vmax = 4, cmap='inferno', 
                    GCaMP_intensity_threshold = 3,
                    acceptable_image_shape_0th_list = [4,32, 33,34],
                    plot_RFP_also = True)
