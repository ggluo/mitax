import sys
sys.path.append('/home/gluo/mitax')

import matplotlib.pyplot as plt
from functools import partial
from mitax.misc import utils
from mitax.mri import ops
import numpy as np 
import cv2
import cmapy
import scipy.stats

savecfl  = lambda p, a: utils.writecfl(p, utils.float2cplx(a)) if a.shape[-1] == 2 else utils.writecfl(p, a)

def subplot(ax, img, title, cmap, interpolation, vmin, vmax):
    ax.imshow(img, cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

plot_params = {'cmap': 'gray', 'interpolation': 'none', 'vmin': 0}
axplot      = partial(subplot, **plot_params)


def prepare_simu(config):
        
    kspace = np.squeeze(np.load(config['ksp_path'])['kspace'])

    nx, ny, _ = kspace.shape
    coilsen = np.squeeze(utils.bart(1, 'ecalib -m1 -r20 -c0.001', kspace[np.newaxis, ...]))
    img_shape = [nx, ny]
    std_coils = ops.mifft2(kspace, img_shape)

    rss = np.sum(np.multiply(std_coils, np.squeeze(np.conj(coilsen))), axis=2)
    mask = utils.bart(1, 'poisson -Y %d -Z %d -y %f -z %f -s 1234 -v -C %d'%(nx, ny, config['fx'], config['fy'], config['cal']))
    mask = np.squeeze(mask)

    und_ksp = kspace*abs(mask[..., np.newaxis])

    coilsen = np.squeeze(utils.bart(1, 'ecalib -m1 -r20 -c0.001', und_ksp[np.newaxis, ...]))
    coilsen = np.squeeze(coilsen)
    x_ = ops.AT_cart(und_ksp, coilsen, mask, img_shape)

    return x_, mask, coilsen, (nx, ny), rss, und_ksp


GRAY2RGB   = lambda x : cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
FLOAT2GRAY = lambda x, exposure_factor=1.0 : np.uint8(x/np.max(x) * 255.* exposure_factor)
GRAY2COLOR = lambda x, name : cv2.applyColorMap(x, cmapy.cmap(name))
BGR2RGB    = lambda x : cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
GBLUR      = lambda x, ksize, sigma : cv2.GaussianBlur(x, ksize, sigma)
def CUT_OFF(x, th):
    x[x<th] = 0
    return x

def mean_confidence_interval(std, confidence=0.95, N=10):
    se = std/np.sqrt(N-1)   # standard error. Since data size is 10 and we compute std with ddof=0, here we use sqrt(N-1)
    h = se * scipy.stats.t.ppf((1+confidence)/2., N-1)
    return h

def fusion(mmse_, var_, colormap, exposure_factor, threshold, ksize, sigma, weight):
    mmse_gray   = FLOAT2GRAY(abs(mmse_))
    mmse_rgb    = GRAY2RGB(abs(mmse_gray))
    var         = FLOAT2GRAY(abs(var_), exposure_factor*np.max(abs(var_)))
    var_color   = BGR2RGB(GRAY2COLOR(var, colormap))
    var_color   = CUT_OFF(var_color, threshold)
    var_color   = GBLUR(var_color, ksize, sigma)
    fusion_mmse = (1.0-weight) * mmse_rgb + weight*var_color

    return var_color, fusion_mmse, mmse_rgb

def soft_fusion(m, v, confidence=0.95, N=10, colormap='terrain', exposure_factor=2, threshold=200, ksize=(5,5), sigma=8, weight=0.1):
    interval = mean_confidence_interval(abs(v), confidence, N)
    return fusion(m, interval, colormap, exposure_factor, threshold, ksize, sigma, weight)

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


def plot_fusion(var_c, err, fusion_m, rgb_m, coil_comb, name):
    plt.figure(figsize=(20, 4))
    plt.subplot(1, 5, 1)
    plt.imshow(var_c)
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(err, cmap='viridis', vmin=0, vmax=0.1)
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(np.uint8(fusion_m))
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(rgb_m)
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(coil_comb)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(name)