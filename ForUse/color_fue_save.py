import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os


def color_fue(savefolder = r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage\ForUse",
              vmin =1, vmax=10, cmap='inferno', label_text = "F/F0",fontsize = 48,
              savefig = True
              ):
    
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    
    # ===== Vertical Right =====
    fig, ax = plt.subplots(figsize=(2, 6), facecolor='none')
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical')
    cbar.ax.set_yticks([])
    cbar.ax.tick_params(size=0)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    
    # Add value labels
    cbar.ax.text(0.5, 1.02, str(vmax), ha='center', va='bottom', fontsize=fontsize, transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.02, str(vmin), ha='center', va='top', fontsize=fontsize, transform=cbar.ax.transAxes)
    
    # Add F/F₀ label to right side
    cbar.ax.text(1.3, 0.5, label_text, ha='left', va='center', fontsize=fontsize, rotation=90, transform=cbar.ax.transAxes)
    
    plt.tight_layout()
    savepath = os.path.join(savefolder, f"vert_rt_{vmin}to{vmax}.png")
    plt.savefig(savepath, dpi = 150, bbox_inches = "tight")
    plt.show()
        
    # ===== Vertical Left =====
    fig, ax = plt.subplots(figsize=(2, 6), facecolor='none')
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical')
    cbar.ax.set_yticks([])
    cbar.ax.tick_params(size=0)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    
    # Value labels
    cbar.ax.text(0.5, 1.02, str(vmax), ha='center', va='bottom', fontsize=fontsize, transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.02, str(vmin), ha='center', va='top', fontsize=fontsize, transform=cbar.ax.transAxes)
    
    # Label on left side
    cbar.ax.text(-0.3, 0.5, label_text, ha='right', va='center', fontsize=fontsize, rotation=90, transform=cbar.ax.transAxes)
    
    plt.tight_layout()
    savepath = os.path.join(savefolder, f"vert_lt_{vmin}to{vmax}.png")
    plt.savefig(savepath, dpi = 150, bbox_inches = "tight")
    plt.show()
    
    
    
    
    # ===== Horizontal =====
    fig, ax = plt.subplots(figsize=(6, 2), facecolor='none')
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.ax.set_xticks([])
    cbar.ax.tick_params(size=0)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    
    # Value labels
    cbar.ax.text(-0.02, 0.5, str(vmin), ha='right', va='center', fontsize=fontsize, transform=cbar.ax.transAxes)
    cbar.ax.text(1.02, 0.5, str(vmax), ha='left', va='center', fontsize=fontsize, transform=cbar.ax.transAxes)
    
    # Label below bar
    cbar.ax.text(0.5, -0.1, label_text, ha='center', va='top', fontsize=fontsize, transform=cbar.ax.transAxes)
    
    plt.tight_layout()
    savepath = os.path.join(savefolder, f"hori_{vmin}to{vmax}.png")
    plt.savefig(savepath, dpi = 150, bbox_inches = "tight")
    plt.show()

if __name__ == "__main__":
    color_fue()