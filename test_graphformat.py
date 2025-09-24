import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as colors
import matplotlib.patheffects as mpe
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from graphformat import set_rc_params


# Fixing random state for reproducibility
np.random.seed(19680801)




fig = plt.figure(figsize=(10/2.5, 2.5))
gs  = fig.add_gridspec(1, 2,  width_ratios=(1, 3),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.5)
# Create the Axes.
ax_scat = fig.add_subplot(gs[1])
ax_hist = fig.add_subplot(gs[0])


# Hexbin
N = 500
x = np.random.rand(N)
y = np.random.rand(N)
z = np.random.rand(N)

extent = (0, 1, 0, 1)
gridsize = (20,15)


viridis = mpl.cm.viridis
newcolors = viridis(np.linspace(0, 1, 9))
newcolors[:1, :] = colors.to_rgba('grey')
newcmp = ListedColormap(newcolors)
norm = colors.Normalize(vmin=6.5, vmax=11)

hb = ax_scat.hexbin(x, y, cmap=newcmp, mincnt=1, gridsize=gridsize, extent=extent, norm=norm)


# Scatter = obs data
N = 150
x_obs = np.random.rand(N)
y_obs = np.random.rand(N)
z_obs = np.random.rand(N)
sc_points = ax_scat.scatter(x_obs, y_obs, s=10, marker='v', alpha=0.75, linewidths=0.4, edgecolor='r', facecolor='r', label='Davis+19')

# Outline formatting
outline=mpe.withStroke(linewidth=1.6, foreground='black')

# Medians
add_median_line = True
if add_median_line:
    #-----------------
    # Define binning parameters
    hist_bins = np.arange(0, 1.1, 0.1)  # Binning edges

    # Compute statistics in each bin
    medians = []
    lower_1sigma = []
    upper_1sigma = []
    bin_centers = []
    bins_n = []
    for i in range(len(hist_bins) - 1):
        mask = (y >= hist_bins[i]) & (y < hist_bins[i + 1])
        y_bin = z[mask]
        
        # Append bin count
        bins_n.append(len(y_bin))
        
        if len(y_bin) >= 1:  # Ensure the bin contains data
            medians.append(np.median(y_bin))
            lower_1sigma.append(np.percentile(y_bin, 25))  # -1σ (25th percentile)
            upper_1sigma.append(np.percentile(y_bin, 75))  # +1σ (75th percentile)
            bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
        else:
            medians.append(math.nan)
            lower_1sigma.append(math.nan)
            upper_1sigma.append(math.nan)
            bin_centers.append((hist_bins[i] + hist_bins[i + 1]) / 2)  # Bin center in log space
            
    # Convert bin centers back to linear space for plotting
    medians = np.array(medians)
    bin_centers = np.array(bin_centers)
    bins_n = np.array(bins_n)
    medians_masked = np.ma.masked_where(bins_n < 10, medians)
    lower_1sigma_masked = np.ma.masked_where(bins_n < 10, lower_1sigma)
    upper_1sigma_masked = np.ma.masked_where(bins_n < 10, upper_1sigma)
    
    
    ax_scat.plot(bin_centers, medians, color='C0', ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline])
    ax_scat.plot(bin_centers, medians_masked, color='C0', linewidth=1, label='ETGs (excl. FRs)', zorder=20, path_effects=[outline])
    ax_scat.plot(bin_centers, lower_1sigma_masked, color='C0', linewidth=0.7, ls='--', zorder=20, path_effects=[outline])
    ax_scat.plot(bin_centers, upper_1sigma_masked, color='C0', linewidth=0.7, ls='--', zorder=20, path_effects=[outline])
    
    ax_scat.plot(bin_centers, medians-0.1, color='C1', ls=(0, (1, 1)), linewidth=1, zorder=10, path_effects=[outline])
    ax_scat.plot(bin_centers, medians_masked-0.1, color='C1', linewidth=1, label='ETGs (incl. FRs)', zorder=20, path_effects=[outline])
    ax_scat.plot(bin_centers, lower_1sigma_masked-0.1, color='C1', linewidth=0.7, ls='--', zorder=20, path_effects=[outline])
    ax_scat.plot(bin_centers, upper_1sigma_masked-0.1, color='C1', linewidth=0.7, ls='--', zorder=20, path_effects=[outline])


# histograms
add_detection_hist_obs = True
if add_detection_hist_obs:
    
    mask_h2 = z_obs >= 0.5
    
    # we want the fraction within a bin, not normalised
    bin_width = 0.1
    hist_bins = np.arange(0, 1.1, bin_width)  # Binning edges
    bin_n, _           = np.histogram(y_obs, bins=hist_bins)
    bin_n_detected, _  = np.histogram(y_obs[mask_h2], bins=hist_bins)
    mask_positive_n     = bin_n > 0
    bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
    
    # returns a % upper and a % lower
    bin_f_detected_err_lower = np.array([0.1, 0.1, 0.1, 0.05, 0.1, 0.05, 0.1, 0.2, 0.1, 0.1])
    bin_f_detected_err_upper = np.array([0.1, 0.1, 0.1, 0.05, 0.1, 0.05, 0.1, 0.2, 0.1, 0.1])
    
    # barh
    #ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), bin_f_detected, height=bin_width, align='edge', orientation='horizontal', fill=False, edgecolor='r', linewidth=1) 
    ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), bin_f_detected, height=bin_width, align='edge', orientation='horizontal', facecolor='r', edgecolor=None, linewidth=1, alpha=0.25, label='Davis+19') 
    
    #ax_hist.axvline(100*f_detected, ls='--', c='grey', lw=0.7)
    
    # errorbar
    hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
    #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
    #ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='r', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.25)
add_detection_hist_ETG = True
if add_detection_hist_ETG:
    
    mask_h2 = z >= 0.5
    
    # we want the fraction within a bin, not normalised
    bin_width = 0.1
    hist_bins = np.arange(0, 1.1, bin_width)  # Binning edges
    bin_n, _           = np.histogram(y, bins=hist_bins)
    bin_n_detected, _  = np.histogram(y[mask_h2], bins=hist_bins)
    mask_positive_n     = bin_n > 0
    bin_f_detected      = bin_n_detected[mask_positive_n]/bin_n[mask_positive_n]
    
    # returns a % upper and a % lower
    bin_f_detected_err_lower = np.array([0.1, 0.1, 0.1, 0.05, 0.1, 0.05, 0.1, 0.2, 0.1, 0.1])
    bin_f_detected_err_upper = np.array([0.1, 0.1, 0.1, 0.05, 0.1, 0.05, 0.1, 0.2, 0.1, 0.1])
    
    
    # barh
    #ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), bin_f_detected, height=bin_width, align='edge', orientation='horizontal', fill=False, edgecolor='C0', linewidth=1) 
    #ax_hist.barh(((hist_bins[:-1])[mask_positive_n]), bin_f_detected-0.1, height=bin_width, align='edge', orientation='horizontal', fill=False, edgecolor='C1', linewidth=1)
    
    # step
    ax_hist.stairs(bin_f_detected, hist_bins[np.append(mask_positive_n, True)], orientation='horizontal', fill=False, baseline=0, edgecolor='C0', linewidth=1, path_effects=[outline])
    ax_hist.stairs(bin_f_detected-0.1, hist_bins[np.append(mask_positive_n, True)], orientation='horizontal', fill=False, baseline=0, edgecolor='C1', linewidth=1, path_effects=[outline])
    
    #ax_hist.axvline(100*f_detected, ls='--', c='grey', lw=0.7)
    
    # errorbar
    hist_bins_midpoint = (hist_bins[:-1][mask_positive_n]) + ((hist_bins[1] - hist_bins[0])/2)
    #mask_positive_n_det = bin_n_detected[mask_positive_n] > 0
    ax_hist.errorbar(bin_f_detected, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C0', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)
    ax_hist.errorbar(bin_f_detected-0.1, hist_bins_midpoint, xerr=[bin_f_detected_err_lower, bin_f_detected_err_upper], yerr=None, ecolor='C1', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=0.7, alpha=0.7)


# colorbar
fig.colorbar(hb, ax=ax_scat, label='$\kappa_{\mathrm{co}}^{*}$', extend='min')      #, extend='max'  


ax_scat.set_xlim(0, 1)
ax_hist.set_ylim(0, 1)
ax_scat.set_ylim(0, 1)
ax_scat.minorticks_on()
ax_hist.minorticks_on()
ax_scat.set_xlabel(r'log$_{10}$ $M_{*}$ ($50$ pkpc) [M$_{\odot}$]')
ax_hist.set_xlabel(r'Normalised freq.')
ax_scat.set_yticklabels([])
ax_hist.set_ylabel(r'$u^{*} - r^{*}$')
    
    
ax_scat.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=False)
ax_hist.legend(ncol=1, frameon=False, scatterpoints = 1, labelspacing=0.1, loc='upper right', handletextpad=0.4, handlelength=0.8, markerfirst=False)

plt.show()



















