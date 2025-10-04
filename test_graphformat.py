import numpy as np
import scipy
import math
import h5py
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
from read_dataset_directories_colibre import _assign_directories


# Fixing random state for reproducibility
np.random.seed(19680801)

#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens\n     3 cosma8           ->  ")
COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir, obs_dir = _assign_directories(answer)
#====================================





fig = plt.figure(figsize=(10/3, 2.8))
gs  = fig.add_gridspec(1, 2,  width_ratios=(1, 1),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.3)
# Create the Axes.
ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1])


ax_top.set_yticklabels([])
#ax_bot.set_xlabel('title')

ax_top.set_ylabel('test')
#ax_bot.set_ylabel('test')
fig.supxlabel('test')


plt.show()

raise Exception('current break 98yhoika')

#=================================================

# Graph initialising and base formatting
fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
plt.subplots_adjust(wspace=0.4, hspace=0.3)

        
add_serra2012    = True
        
if add_serra2012:      # ($z=0.0$)
    # Load the observational data
    with h5py.File('%s/Serra2012_ATLAS3D_HI.hdf5'%obs_dir, 'r') as file:
        obs_names_1 = file['data/Galaxy/values'][:]
        obs_HI       = file['data/log_H1/values'][:] 
        obs_mask_1  = file['data/det_mask/values'][:]

    with h5py.File('%s/Cappellari2011_masses.hdf5'%obs_dir, 'r') as file:
        obs_names_2 = file['data/Galaxy/values'][:]
        obs_mstar  = file['data/log_Mstar/values'][:] 
            
    obs_names_1 = np.array(obs_names_1)
    obs_mask_1  = np.array(obs_mask_1, dtype=bool)
    obs_names_1 = obs_names_1[obs_mask_1]
    obs_HI      = obs_HI[obs_mask_1]
            
    obs_names_2 = np.array(obs_names_2)
            
    # Match galaxy names to get mass (log)
    obs_x = []
    for name_i in obs_names_1:
        mask_name = np.argwhere(name_i == obs_names_2).squeeze()
        obs_x.append(obs_mstar[mask_name])
                
    obs_x = np.array(obs_x)
    assert len(obs_x) == len(obs_HI), 'Some galaxy names unmatched...? x: %s y: %s'%(len(obs_x), len(obs_HI))
    
    

    #obs_y = obs_HI - np.log10((10**obs_HI) + (10**obs_x))
    obs_y = obs_HI
            
    print('Sample length Serra+12 ATLAS3D:   %s'%len(obs_y))
        

    

# Histograms   
hist_bin_width = 0.2
lower_mass_limit = 10**6
upper_mass_limit = 10**11
box_volume = 1.16e5
        
    
hist_masses, bin_edges =  np.histogram(obs_y, bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
hist_masses = hist_masses[:]/(box_volume)      # in units of /cMpc**3
hist_masses = hist_masses/hist_bin_width        # density
bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        
# Add poisson errors to each bin (sqrt N)
hist_n, _ = np.histogram(obs_y, bins=np.arange(np.log10(lower_mass_limit), np.log10(upper_mass_limit)+hist_bin_width, hist_bin_width))
hist_err = (np.sqrt(hist_n)/(box_volume))/hist_bin_width

# Masking out nans
with np.errstate(divide='ignore', invalid='ignore'):
    hist_mask_finite = np.isfinite(np.log10(hist_masses))
hist_masses = hist_masses[hist_mask_finite]
bin_midpoints   = bin_midpoints[hist_mask_finite]
hist_err    = hist_err[hist_mask_finite]
hist_n      = hist_n[hist_mask_finite]

#axs.plot(np.flip(bin_midpoints), np.flip(hist_masses), color='k', ls='-', zorder=-4)
axs.errorbar(np.flip(bin_midpoints), np.flip(hist_masses), yerr=np.flip(hist_err))
        

plt.xlim(6, 11)
plt.ylim(10**(-6), 10**(-1))
#plt.xscale("log")
plt.yscale("log")

plt.show()
plt.close()


raise Exception('current break 9y1ohuj')
##################################################################################################################################



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



















