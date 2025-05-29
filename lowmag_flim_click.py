#%%
import os, glob
import datetime
from multi_z_stack_functions import click_zstack_flimfile

def click_flim_list(folder, filename_example):
    filelist = glob.glob(os.path.join(folder,filename_example))
    for each_flim in filelist:
        now = datetime.datetime.now()
        modified_date = datetime.datetime.fromtimestamp(os.path.getmtime(each_flim))
        delta = (now - modified_date).total_seconds()
        if delta > 20:
            click_zstack_flimfile(each_flim, use_predefined_pos = True)
            
    for each_flim in filelist:
        print('r"'+each_flim+'",')


if __name__ == "__main__":
    folder = r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20250520\E4bufferMNI4Ca"
    filename_example = "lowmag*_001.flim"
    click_flim_list(folder, filename_example)
# %%
