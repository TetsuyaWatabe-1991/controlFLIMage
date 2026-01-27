import sys
sys.path.append("..\\..")
from flimage_graph_func import plot_GCaMP_F_F0

#select file dialog
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(filetypes=[("Flim files", "*.flim")])
root.destroy()

plot_GCaMP_F_F0(file_path, GCaMP_ch_1or2 = 2,
                vmin = 1, vmax = 4, cmap='inferno', 
                acceptable_image_shape_0th_list = [4,32, 33,34])