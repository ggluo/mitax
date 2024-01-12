import sys
sys.path.append('/home/gluo/mitax')

import matplotlib.pyplot as plt
from functools import partial
from mitax.misc import utils
import numpy as np 

savecfl  = lambda p, a: utils.writecfl(p, utils.float2cplx(a)) if a.shape[-1] == 2 else utils.writecfl(p, a)

def subplot(ax, img, title, cmap, interpolation, vmin, vmax):
    ax.imshow(img, cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

plot_params = {'cmap': 'gray', 'interpolation': 'none', 'vmin': 0}
axplot      = partial(subplot, **plot_params)


def plot_grid(grid_x, grid_y, image_n):
    
    images=np.abs(utils.float2cplx(image_n))
    fig, axss = plt.subplots(grid_x, grid_y, figsize=(10, 10), gridspec_kw={'width_ratios': [1  for _ in range(grid_x)]})
    for i in range(grid_x):
        for j in range(grid_y):
            if i==0:
                strs='x_%d'%j
            else:
                strs=''
            axplot(axss[i,j], images[i*grid_y+j], title=strs, vmax=np.max(images[i*grid_y+j]))
    plt.tight_layout(pad=0.)