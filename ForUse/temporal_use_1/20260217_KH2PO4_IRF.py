import os
import sys
import glob
sys.path.append("..\..")
from FLIMageFileReader2 import FileReader
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, amplitude, center, width):
    return amplitude * np.exp(-(x - center)**2 / (2 * width**2))


file_path = r"G:\ImagingData\Tetsuya\20260317\KH2PO4_001.flim"

filelist = glob.glob(file_path[:-9] + '*.flim')
print(filelist)

#change order of filelist by os.path.getmtime
filelist.sort(key=lambda x: os.path.getmtime(x))

savefolder = r"G:\ImagingData\Tetsuya\20260317\KH2PO4_920ex_IRF"
os.makedirs(savefolder, exist_ok=True)

IRF_savefolder = os.path.join(savefolder, "IRF")
os.makedirs(IRF_savefolder, exist_ok=True)

GFP_2d_savefolder = os.path.join(savefolder, "GFP_2d")
os.makedirs(GFP_2d_savefolder, exist_ok=True)

#get earliest acquisition time using os.path.getmtime
earliest_acquisition_time = min(filelist, key=lambda x: os.path.getmtime(x))
print(os.path.getmtime(earliest_acquisition_time))


for file_path in filelist:
    ch1or2 = 1
    iminfo = FileReader()
    iminfo.read_imageFile(file_path, True)     
    six_dim = np.array(iminfo.image)
    sync1_threshold = iminfo.statedict['State.Spc.spcData.sync_threshold'][0]

    savefolder_2 = os.path.join(savefolder, f'thresh_{sync1_threshold}mV')
    os.makedirs(savefolder_2, exist_ok=True)
    
    elapsed_time = os.path.getmtime(file_path) - os.path.getmtime(earliest_acquisition_time)
    elapsed_time_min = int(elapsed_time // 60)


    GFP_ch_lifetime=six_dim[0,0,ch1or2-1,:,:,:].sum(axis=-2).sum(axis=-2)

    time_ns = np.arange(0, len(GFP_ch_lifetime)*12.5/64, 12.5/64)
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

#gaussian fit to the IRF

    popt, pcov = curve_fit(gaussian, time_ns, GFP_ch_lifetime)
    amplitude, center, width = popt


    each_bin_ns = 12.5/64    
    plt.figure(figsize=(4, 2))
    plt.title(f'{elapsed_time_min} min, sync1_threshold: {sync1_threshold} mV')
    plt.plot(time_ns, GFP_ch_lifetime, label='Data')
    plt.plot(time_ns, gaussian(time_ns, amplitude, center, width), 'r-', label='Gaussian fit')
    plt.legend()

    if GFP_ch_lifetime.max()>0:
        plt.scatter(second_peak_time, second_peak_intensity, c='r', marker='o')
        plt.scatter(peak_time, peak_intensity, c='g', marker='o')
        ratio =  peak_intensity / second_peak_intensity
        plt.text(peak_time, peak_intensity, f'x{ratio:.2f}', color='g', fontsize=12,
                horizontalalignment='left', verticalalignment='top')

    #log scale
    plt.yscale('log')
    plt.xlabel('Time (ns)')
    plt.ylabel('Intensity')
    plt.ylim(basal_intensity*0.5, peak_intensity*1.2)

    plt.savefig(os.path.join(savefolder, os.path.basename(file_path)[:-5] + '_IRF.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(savefolder_2, os.path.basename(file_path)[:-5] + '_IRF.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(IRF_savefolder, os.path.basename(file_path)[:-5] + '_IRF.png'), dpi=150, bbox_inches='tight')
    plt.show()

    GFP_2dimg = six_dim[0,0,ch1or2-1,:,:,:].sum(axis=-1)
    plt.figure(figsize=(3, 3))
    plt.imshow(GFP_2dimg, cmap='gray')
    plt.axis('off')
    plt.title(f'{elapsed_time_min} min, sync1: {sync1_threshold} mV')
    plt.savefig(os.path.join(savefolder, os.path.basename(file_path)[:-5] + '_GFP_2d.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(savefolder_2, os.path.basename(file_path)[:-5] + '_GFP_2d.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(GFP_2d_savefolder, os.path.basename(file_path)[:-5] + '_GFP_2d.png'), dpi=150, bbox_inches='tight')
    plt.show()
    # print(np.array(iminfo.image).shape)

#     break