# %%
import os
import sys
sys.path.append("..\..")
from FLIMageFileReader2 import FileReader
import numpy as np
from custom_plot import plt
from PIL import Image
filepath = r"G:\ImagingData\Tetsuya\20260120\ASAP5_ori_2ndpos_002.flim"

iminfo = FileReader()
iminfo.read_imageFile(filepath, True)
six_dim = np.array(iminfo.image)
print(np.array(iminfo.image).shape)

ch1or2 = 1

GFPimg=six_dim[:,0,ch1or2-1,:,:,:].sum(axis=-1)

pixel_line_time_ms = 2
pixel_per_line = 128


# %%

print("image shape: ", GFPimg.shape)

plt.imshow(GFPimg[0,:,:], cmap="gray")
plt.axis("off")
plt.show()

#each frame is stitched vertically, from first to the last frame (100th frame)

stitched_GFPimg = None
for i in range(100):
    if stitched_GFPimg is None:
        stitched_GFPimg = GFPimg[i,:,:]
    else:
        stitched_GFPimg = np.concatenate((stitched_GFPimg, GFPimg[i,:,:]), axis=0)

# %%

pixels_pre_additional = 40
pixels_add_after = 70

nth_uncaging = 0
t_axis_from = (nth_uncaging+1)*10*128 - pixels_pre_additional
t_axis_to = t_axis_from + 128 + pixels_add_after
frame_from = t_axis_from / 128
frame_to = t_axis_to / 128
uncaging_img = stitched_GFPimg[t_axis_from:t_axis_to,:]
print(frame_from, frame_to)

baseline_img = stitched_GFPimg[0:128+pixels_add_after,:]

plt.imshow(baseline_img, cmap="inferno")
plt.axis("off")
plt.show()
plt.imshow(uncaging_img, cmap="inferno")
plt.axis("off")
plt.show()

#save as tiff
savepath = filepath[:-5] + "_uncaging_img.tiff"
Image.fromarray(uncaging_img).save(savepath)
print("uncaging_img saved to ", savepath)


# %%
scale_ms = 100
scale_pixel_len = scale_ms/pixel_line_time_ms

nth_uncaging = 0
total_uncaging_num = 18
plt.subplots(2, 5, figsize=(5*5, 5*2))
vmax = np.max(baseline_img)
vmin = np.min(baseline_img)

scale_y_pos = 30

ax = plt.subplot(2, 5, 1)
ax.imshow(np.rot90(baseline_img, k=1, axes=(0,1)), cmap="inferno", vmax=vmax, vmin=vmin)
ax.axis("off")
ax.plot([10,10+scale_pixel_len], [scale_y_pos, scale_y_pos], color="white", linewidth=2)
ax.text(10+scale_pixel_len/2, scale_y_pos, f"{scale_ms} ms", color="white", fontsize=12, ha="center", va="bottom")   
ax.set_title(f"baseline")
#get current vmax and vmin

for nth_uncaging in range(total_uncaging_num):
    t_axis_from = (nth_uncaging+1)*10*128 - pixels_pre_additional
    t_axis_to = t_axis_from + 128 + pixels_add_after
    print(t_axis_from, t_axis_to,"difference: ", t_axis_to - t_axis_from)
    img = stitched_GFPimg[t_axis_from:t_axis_to,:]
    print("img.shape: ", img.shape)

    if nth_uncaging < 9:
        ax = plt.subplot(2, 5, nth_uncaging+2)
        ax.imshow(np.rot90(img, k=1, axes=(0,1)), cmap="inferno", vmax=vmax, vmin=vmin)
        ax.axis("off")
        ax.set_title(f"uncaging {nth_uncaging+1}")
    if nth_uncaging == 0:
        averaged = img.copy()
        print("averaged initialized")
    else:
        averaged += img
savepath = filepath[:-5] + "_baseline_and_uncaging_plt_all.png"
plt.savefig(savepath, dpi=150, bbox_inches="tight")
print("baseline_and_uncaging_plt_all saved to ", savepath)
plt.show()

plt.imshow(averaged, cmap="inferno", vmax=vmax, vmin=vmin)
plt.axis("off")
plt.show()

savepath = filepath[:-5] + "_averaged_uncaging_img.tiff"
Image.fromarray(averaged).save(savepath)
print("averaged_uncaging_img saved to ", savepath)

# %%
#rotate 90 degrees and plot averaged
plt.imshow(np.rot90(averaged, k=1, axes=(0,1)), cmap="inferno")
plt.axis("off")
plt.plot([10,10+scale_pixel_len], [scale_y_pos, scale_y_pos], color="white", linewidth=2)
plt.text(10+scale_pixel_len/2, scale_y_pos, f"{scale_ms} ms", color="white", fontsize=12, ha="center", va="bottom")   
savepath = filepath[:-5] + "_averaged_uncaging_img_rot90.png"
plt.savefig(savepath, dpi=150, bbox_inches="tight")
print("averaged_uncaging_img_rot90 saved to ", savepath)
plt.show()

# %%