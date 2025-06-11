import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

def draw_boundaries(mask, color, linewidth = 0.5):
    boundaries = find_boundaries(mask, mode='thick')
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if boundaries[y, x]:
                if y == 0 or mask[y-1, x] != mask[y, x]:
                    plt.plot([x, x+1], [y, y], color=color, linewidth=linewidth)
                if y == mask.shape[0]-1 or mask[y+1, x] != mask[y, x]:
                    plt.plot([x, x+1], [y+1, y+1], color=color, linewidth=linewidth)
                if x == 0 or mask[y, x-1] != mask[y, x]:
                    plt.plot([x, x], [y, y+1], color=color, linewidth=linewidth)
                if x == mask.shape[1]-1 or mask[y, x+1] != mask[y, x]:
                    plt.plot([x+1, x+1], [y, y+1], color=color, linewidth=linewidth)