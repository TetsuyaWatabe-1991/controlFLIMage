import sys
sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")
import glob
import os
import datetime
import pandas as pd
from FLIMageFileReader2 import FileReader
import matplotlib.pyplot as plt
import numpy as np

one_of_filepath = r"G:\ImagingData\Tetsuya\20250903\lowmag\B1cnt_001.flim"
file_list = glob.glob(os.path.join(os.path.dirname(one_of_filepath), "*.flim"))

for nth, each_filepath in enumerate(file_list):
    iminfo = FileReader()
    iminfo.read_imageFile(each_filepath, readImage = True)
    six_dim = np.array(iminfo.image)
    z_projection = six_dim[:,:,1,:,:].sum(axis=1).sum(axis=-1).max(axis=0)
    plt.imshow(z_projection, cmap="gray")
    plt.axis("off")
    savepath = each_filepath.replace(".flim", ".png")
    plt.savefig(savepath, dpi=150,bbox_inches="tight")
    plt.show()

