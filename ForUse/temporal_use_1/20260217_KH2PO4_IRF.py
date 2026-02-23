import os
import sys
import glob
sys.path.append("..\..")
from FLIMageFileReader2 import FileReader
import numpy as np
import matplotlib.pyplot as plt

file_path = r"G:\ImagingData\Tetsuya\20260217\KH2PO4_920ex_filterHQ_430_averaged_001.flim"

filelist = glob.glob(file_path[:-9] + '*.flim')
print(filelist)

savefolder = r"G:\ImagingData\Tetsuya\20260217\KH2PO4_920ex_filterHQ_430_averaged_IRF"
os.makedirs(savefolder, exist_ok=True)

for file_path in filelist:
    ch1or2 = 1
    iminfo = FileReader()
    iminfo.read_imageFile(file_path, True)     
    six_dim = np.array(iminfo.image)
    sync1_threshold = iminfo.statedict['State.Spc.spcData.sync_threshold'][0]

    GFP_ch_lifetime=six_dim[0,0,ch1or2-1,:,:,:].sum(axis=-2).sum(axis=-2)

    find_peak_pos = np.argmax(GFP_ch_lifetime)
    peak_intensity = GFP_ch_lifetime[find_peak_pos]
    peak_time = time_ns[find_peak_pos]

    second_peak_find_area_1 = GFP_ch_lifetime[max(0, find_peak_pos - 15):find_peak_pos-4]
    second_peak_find_area_2 = GFP_ch_lifetime[find_peak_pos+7:min(len(GFP_ch_lifetime), find_peak_pos+19)]

    second_peak_pos_1 = np.argmax(second_peak_find_area_1) + max(0, find_peak_pos - 15)
    second_peak_pos_2 = np.argmax(second_peak_find_area_2) + find_peak_pos+7
    second_peak_intensity_1 = GFP_ch_lifetime[second_peak_pos_1]
    second_peak_intensity_2 = GFP_ch_lifetime[second_peak_pos_2]
    second_peak_time_1 = time_ns[second_peak_pos_1]
    second_peak_time_2 = time_ns[second_peak_pos_2]

    second_peak_intensity = max(second_peak_intensity_1, second_peak_intensity_2)
    second_peak_time = second_peak_time_1 if second_peak_intensity_1 > second_peak_intensity_2 else second_peak_time_2

    basal_intensity = np.percentile(GFP_ch_lifetime, 10)


    each_bin_ns = 12.5/64
    time_ns = np.arange(0, len(GFP_ch_lifetime)*each_bin_ns, each_bin_ns)
    plt.figure(figsize=(10, 5))
    plt.title(f'sync1_threshold: {sync1_threshold} mV')
    plt.plot(time_ns, GFP_ch_lifetime)
    plt.scatter(second_peak_time, second_peak_intensity, c='r', marker='o')
    plt.scatter(peak_time, peak_intensity, c='g', marker='o')
    ratio =  peak_intensity / second_peak_intensity
    plt.text(peak_time, peak_intensity, f'x{ratio:.2f}', color='g', fontsize=12,
            horizontalalignment='right', verticalalignment='bottom')
    #log scale
    plt.yscale('log')
    plt.xlabel('Time (ns)')
    plt.ylabel('Intensity')
    plt.grid(True)
    plt.ylim(basal_intensity*0.5, peak_intensity*1.1)
    plt.savefig(os.path.join(savefolder, os.path.basename(file_path)[:-5] + '_IRF.png'), dpi=150, bbox_inches='tight')
    plt.show()

    GFP_2dimg = six_dim[0,0,ch1or2-1,:,:,:].sum(axis=-1)
    plt.figure(figsize=(5, 5))
    plt.imshow(GFP_2dimg, cmap='gray')
    plt.axis('off')
    plt.title(f'sync1_threshold: {sync1_threshold} mV')
    plt.savefig(os.path.join(savefolder, os.path.basename(file_path)[:-5] + '_GFP_2d.png'), dpi=150, bbox_inches='tight')
    plt.show()
    # print(np.array(iminfo.image).shape)