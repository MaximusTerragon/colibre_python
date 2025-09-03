import h5py
import numpy as np
import pandas as pd
import math
from read_dataset_directories_colibre import _assign_directories

#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens\n     3 cosma8           ->  ")
COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir, obs_dir = _assign_directories(answer)
#====================================


"""
Online resource: https://plotdigitizer.com/app

"""


#--------------
# GSMF
def _create_Driver2022_new():
    # Create and write
    with h5py.File("%s/GalaxyStellarMassFunction/Driver2022_complete.hdf5"%obs_dir, "a") as f:
        
        # Creating metadata
        grp = f.create_group("metdata")
        grp.attrs["bibcode"]      = "2022MNRAS.513..439D"
        grp.attrs["citation"]     = "Driver et al. (2022) (GAMA-DR4)"
        grp.attrs["comment"]      = "Data obtained assuming a Chabrier IMF and h = 0.7. h-corrected for ** COLIBRE by me ** using cosmology: h=0.681, and includes correction down to z=0. Ignoring the mass bins for which GAMA is systematically incomplete."
        grp.attrs["name"]         = "GSMF from GAMA-DR4"
        grp.attrs["plot_as"]      = "points" 
        grp.attrs["redshift"]     = 0.0
        grp.attrs["redshift_lower"] = 0.0
        grp.attrs["redshift_upper"] = 0.0
        
        # Creating cosmology metadata
        grp = f.create_group("cosmology")
        grp.attrs["H0"]         = 68.1
        grp.attrs["Neff"]       = 3.046
        grp.attrs["Ob0"]        = 0.0486
        grp.attrs["Ode0"]       = 0.693922
        grp.attrs["Om0"]        = 0.306
        grp.attrs["Tcmb0"]      = 2.7255
        grp.attrs["m_nu"]       = np.array([0.0, 0.0, 0.0])
        grp.attrs["m_nu_units"] = 'eV'
        grp.attrs["name"]       = 'Abbott22'
        
        #-------------------------------
        # Creating dataset + cosmology
        
        #-----------
        # y-values: total
        grp = f.create_group("data/total/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Galaxy Stellar Mass'
        grp = f.create_group("data/total/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (GSMF)'
        
        # Data
        log_x      = np.arange(6.875, 11.626, 0.250)
        log_y      = 0.0807 + np.array([-0.691, -1.084, -1.011, -1.349, -1.287, -1.544, -1.669, -1.688, -1.795, -1.886, -2.055, -2.142, -2.219, -2.274, -2.292, -2.361, -2.561, -2.922, -3.414, -4.704])
        log_y_scat          = np.array([ 0.176,  0.125,  0.071,  0.092,  0.079,  0.071,  0.045,  0.032,  0.024,  0.020,  0.014,  0.010,  0.009,  0.009,  0.009,  0.010,  0.013,  0.019,  0.032,  0.138])
        lower = (10**log_y) - (10**(log_y - log_y_scat))
        upper = (10**(log_y + log_y_scat)) - (10**log_y)
        y_scat = np.stack((lower, upper), axis=0)
        
        # Adjust for cosmology
        x      = (10**log_x) * (0.70/0.681)**2
        y      = (10**log_y) * (0.70/0.681)**3
        y_scat = y_scat * (0.70/0.681)**3
            
        # Create dataset
        dset = f.create_dataset("data/total/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/total/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/total/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        #-----------
        # y-values: early type morphology
        grp = f.create_group("data/etg/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Galaxy Stellar Mass'
        grp = f.create_group("data/etg/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (GSMF) of galaxies with early-type morphology'
        
                # Data
        log_x      = np.arange(8.625, 11.626, 0.250)
        log_y      = 0.0866 + np.array([-4.022, -3.157, -3.334, -3.491, -3.323, -3.202, -3.077, -3.008, -2.992, -3.074, -3.164, -3.483, -4.623])
        log_y_scat          = np.array([ 0.301,  0.097,  0.087,  0.075,  0.041,  0.035,  0.031,  0.029,  0.029,  0.031,  0.035,  0.048,  0.146])
        lower = (10**log_y) - (10**(log_y - log_y_scat))
        upper = (10**(log_y + log_y_scat)) - (10**log_y)
        y_scat = np.stack((lower, upper), axis=0)
        
        # Adjust for cosmology
        x      = (10**log_x) * (0.70/0.681)**2
        y      = (10**log_y) * (0.70/0.681)**3
        y_scat = y_scat * (0.70/0.681)**3
            
        # Create dataset
        dset = f.create_dataset("data/etg/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/etg/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/etg/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        #-----------
        # y-values: disky morphology
        grp = f.create_group("data/ltg/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Galaxy Stellar Mass'
        grp = f.create_group("data/ltg/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (GSMF) of galaxies with discy morphology'
        
        # Data
        log_x      = np.arange(6.625, 11.126, 0.250)
        log_y      = 0.0866 + np.array([-1.080, -1.098, -0.938, -0.838, -1.348, -1.591, -1.919, -1.837, -1.878, -2.129, -2.222, -2.401, -2.537, -2.824, -3.039, -3.318, -3.766, -4.322, -4.845])
        log_y_scat          = np.array([ 0.301,  0.222,  0.114,  0.077,  0.089,  0.074,  0.071,  0.043,  0.030,  0.026,  0.019,  0.016,  0.017,  0.024,  0.031,  0.041,  0.067,  0.114,  0.222])
        lower = (10**log_y) - (10**(log_y - log_y_scat))
        upper = (10**(log_y + log_y_scat)) - (10**log_y)
        y_scat = np.stack((lower, upper), axis=0)
        
        # Adjust for cosmology
        x      = (10**log_x) * (0.70/0.681)**2
        y      = (10**log_y) * (0.70/0.681)**3
        y_scat = y_scat * (0.70/0.681)**3
            
        # Create dataset
        dset = f.create_dataset("data/ltg/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/ltg/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/ltg/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        print("Successfully created: %s/GalaxyStellarMassFunction/Driver2022_complete.hdf5"%obs_dir)
#--------------
def _create_Kelvin2014():
    # Create and write
    with h5py.File("%s/GalaxyStellarMassFunction/Kelvin2014.hdf5"%obs_dir, "a") as f:
        
        # Creating metadata
        grp = f.create_group("metadata")
        grp.attrs["bibcode"]      = "2014MNRAS.444.1647K"
        grp.attrs["citation"]     = "Kelvin et al. (2014) (GAMA-DR1)"
        grp.attrs["comment"]      = "Data obtained assuming a Chabrier IMF and h = 0.7. h-corrected for ** COLIBRE by me ** using cosmology: h=0.681, and includes correction down to z=0. Ignoring the mass bins for which GAMA is systematically incomplete."
        grp.attrs["name"]         = "GSMF from GAMA-DR1"
        grp.attrs["plot_as"]      = "points" 
        grp.attrs["redshift"]     = 0.0
        grp.attrs["redshift_lower"] = 0.0
        grp.attrs["redshift_upper"] = 0.0
        
        # Creating cosmology metadata
        grp = f.create_group("cosmology")
        grp.attrs["H0"]         = 68.1
        grp.attrs["Neff"]       = 3.046
        grp.attrs["Ob0"]        = 0.0486
        grp.attrs["Ode0"]       = 0.693922
        grp.attrs["Om0"]        = 0.306
        grp.attrs["Tcmb0"]      = 2.7255
        grp.attrs["m_nu"]       = np.array([0.0, 0.0, 0.0])
        grp.attrs["m_nu_units"] = 'eV'
        grp.attrs["name"]       = 'Abbott22'
        
        #-------------------------------
        # Creating dataset + cosmology
        
        
        #-----------
        # y-values: ETG plus little blue spheroids
        grp = f.create_group("data/etg_plus_lbs/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Galaxy Stellar Mass of ETGs including little blue spheroids'
        grp = f.create_group("data/etg_plus_lbs/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (GSMF)'
        
        # Data
        log_x      = np.array([8.2523, 8.3505, 8.4518, 8.5506, 8.6493, 8.7491, 8.8493, 8.9506, 9.0479, 9.1507, 9.2519, 9.3496, 9.4524, 9.5514, 9.6517, 9.7520, 9.8524, 9.9517, 10.0518, 10.1513, 10.2521, 10.3525, 10.4537, 10.5521, 10.6540, 10.7534, 10.8511, 10.9521, 11.0534, 11.1546, 11.2525, 11.3550, 11.4533])
        y          = np.array([4.7614e-05, 0.00034391, 0.00054860, 0.0010827, 0.0015224, 0.0015757, 0.0019260, 0.0021491, 0.0022389, 0.0019188, 0.0016852, 0.0022021, 0.0019533, 0.0020185, 0.0018772, 0.0019037, 0.0024307, 0.0027640, 0.0026351, 0.0026703, 0.0032626, 0.0033716, 0.0046404, 0.0036393, 0.0030433, 0.0021983, 0.0022464, 0.0016224, 0.0014129, 0.00066731, 0.00047775, 0.00023947, 4.7918e-05])
        y_upper_y    = np.array([9.5018e-05, 0.00047325, 0.00070355, 0.0013023, 0.0018041, 0.0018358, 0.0022173, 0.0025139, 0.0025730, 0.0022294, 0.0019721, 0.0025234, 0.0022404, 0.0023314, 0.0021730, 0.0021831, 0.0027511, 0.0031754, 0.0029976, 0.0030709, 0.0036262, 0.0037502, 0.0051079, 0.0039939, 0.0034393, 0.0025623, 0.0025874, 0.0019495, 0.0016609, 0.00087361, 0.00062822, 0.00034298, 9.4481e-05])
        y_lower_y    = np.array([2.4423e-05, 0.00021307, 0.00039134, 0.00085652, 0.0012481, 0.0013048, 0.0016248, 0.0018147, 0.0019031, 0.0016160, 0.0013954, 0.0018676, 0.0016340, 0.0017047, 0.0015680, 0.0015905, 0.0020833, 0.0023863, 0.0022796, 0.0022733, 0.0028755, 0.0029349, 0.0041808, 0.0032111, 0.0026305, 0.0018599, 0.0019042, 0.0013437, 0.0011991, 0.00048147, 0.00032379, 0.00013043, 2.2605e-05])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        
        # Adjust for cosmology
        x      = (10**log_x) * (0.70/0.681)**2
        y      = y * (0.70/0.681)**3
        y_scat = y_scat * (0.70/0.681)**3
            
        # Create dataset
        dset = f.create_dataset("data/etg_plus_lbs/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/etg_plus_lbs/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/etg_plus_lbs/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        
        #-----------
        # y-values: LTG 
        grp = f.create_group("data/ltg/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Galaxy Stellar Mass of disc-dominated galaxies'
        grp = f.create_group("data/ltg/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (GSMF)'
        
        # Data
        log_x      = np.array([8.0495, 8.1489, 8.2497, 8.3512, 8.4506, 8.5494, 8.6521, 8.7499, 8.8512, 8.9507, 9.0508, 9.1533, 9.2494, 9.3503, 9.4517, 9.5502, 9.6511, 9.7527, 9.8524, 9.9514, 10.0545, 10.1517, 10.2520, 10.3513, 10.4530, 10.5531, 10.6527, 10.7531, 10.8526, 10.9531, 11.0527])
        y          = np.array([0.00011535, 0.00021782, 0.00080060, 0.0012954, 0.0041227, 0.0056334, 0.0069398, 0.0084666, 0.0095890, 0.0091491, 0.0086275, 0.0082589, 0.0087465, 0.0072058, 0.0062965, 0.0062897, 0.0044314, 0.0046205, 0.0039598, 0.0035704, 0.0026603, 0.0030454, 0.0029052, 0.0021108, 0.0016286, 0.0010511, 0.0012986, 0.00066833, 0.00038625, 0.00038416, 0.00019172])
        y_upper_y    = np.array([0.00018769, 0.00031851, 0.00099362, 0.0015315, 0.0047608, 0.0062231, 0.0074579, 0.0089904, 0.010219, 0.0097741, 0.0093353, 0.0089462, 0.0095706, 0.0077410, 0.0067943, 0.0068208, 0.0048637, 0.0051001, 0.0043370, 0.0039532, 0.0030014, 0.0034391, 0.0033716, 0.0023930, 0.0018875, 0.0012730, 0.0015386, 0.00084945, 0.00050665, 0.00051488, 0.00028562])
        y_lower_y    = np.array([4.0691e-05, 0.00011566, 0.00060044, 0.0010441, 0.0036339, 0.0051065, 0.0062941, 0.0077437, 0.0088650, 0.0083148, 0.0079424, 0.0075971, 0.0080321, 0.0065633, 0.0056980, 0.0056666, 0.0039512, 0.0041355, 0.0034714, 0.0031613, 0.0023318, 0.0026490, 0.0025314, 0.0017899, 0.0013390, 0.00082654, 0.0010384, 0.00052017, 0.00024769, 0.00024625, 9.4878e-05])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        
        # Adjust for cosmology
        x      = (10**log_x) * (0.70/0.681)**2
        y      = y * (0.70/0.681)**3
        y_scat = y_scat * (0.70/0.681)**3
            
        # Create dataset
        dset = f.create_dataset("data/ltg/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/ltg/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/ltg/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        #-----------
        # y-values: total 
        grp = f.create_group("data/total/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Galaxy Stellar Mass'
        grp = f.create_group("data/total/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (GSMF)'
        
        # Data
        log_x      = np.array([8.0470, 8.1491, 8.2490, 8.3511, 8.4489, 8.5490, 8.6497, 8.7494, 8.8495, 8.9488, 9.0482, 9.1496, 9.2499, 9.3499, 9.4512, 9.5504, 9.6501, 9.7505, 9.8533, 9.9514, 10.0517, 10.1519, 10.2525, 10.3526, 10.4542, 10.5513, 10.6540, 10.7512, 10.8504, 10.9498, 11.0533, 11.1509, 11.2533, 11.3474, 11.4485])
        y          = np.array([0.00011617, 0.00026878, 0.00085626, 0.0016338, 0.0046905, 0.0067443, 0.0085648, 0.010060, 0.011574, 0.011607, 0.010801, 0.010342, 0.010606, 0.0093815, 0.0082090, 0.0083403, 0.0064105, 0.0066110, 0.0063195, 0.0065175, 0.0053107, 0.0058054, 0.0062167, 0.0055491, 0.0063919, 0.0047088, 0.0043853, 0.0028315, 0.0026247, 0.0019791, 0.0016315, 0.00067091, 0.00046248, 0.00024150, 0.00004772])
        y_upper_y    = np.array([0.00018322, 0.00037372, 0.0010433, 0.0019014, 0.0051565, 0.0072732, 0.0090326, 0.010648, 0.012217, 0.012605, 0.011802, 0.011093, 0.011429, 0.010161, 0.0087980, 0.0089211, 0.0069453, 0.0070515, 0.0069286, 0.0069566, 0.0058039, 0.0062767, 0.0067274, 0.0059896, 0.0069337, 0.0051319, 0.0048089, 0.0032065, 0.0029488, 0.0022957, 0.0019029, 0.00087305, 0.00062630, 0.00033668, 0.00009515])
        y_lower_y    = np.array([4.2482e-05, 0.00015200, 0.00064868, 0.0012123, 0.0041012, 0.0057783, 0.0077289, 0.0094501, 0.010847, 0.010695, 0.0099878, 0.0095790, 0.0097726, 0.0086881, 0.0077060, 0.0077621, 0.0057689, 0.0060122, 0.0058301, 0.0059070, 0.0047600, 0.0054678, 0.0056566, 0.0049646, 0.0057505, 0.0041795, 0.0039424, 0.0024198, 0.0023656, 0.0016680, 0.0013929, 0.00050894, 0.00033185, 0.00013281, 2.0211e-05])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        
        # Adjust for cosmology
        x      = (10**log_x) * (0.70/0.681)**2
        y      = y * (0.70/0.681)**3
        y_scat = y_scat * (0.70/0.681)**3
            
        # Create dataset
        dset = f.create_dataset("data/total/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/total/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/total/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        #-----------
        # y-values: total 
        grp = f.create_group("data/etg/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Galaxy Stellar Mass'
        grp = f.create_group("data/etg/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (GSMF)'
        
        # Data
        log_x      = np.array([8.8451, 8.9478, 9.0477, 9.1474, 9.2477, 9.3463, 9.4469, 9.5482, 9.6489, 9.7469, 9.8484, 9.9507, 10.049, 10.151, 10.252, 10.352, 10.453, 10.552, 10.651, 10.753, 10.854, 10.954, 11.056, 11.154, 11.251, 11.355, 11.457])
        y          = np.array([0.00014470, 0.00024172, 0.00036293, 0.00049595, 0.0011275, 0.0016990, 0.0017333, 0.0014795, 0.0017604, 0.0018388, 0.0023696, 0.0027501, 0.0026527, 0.0027129, 0.0032973, 0.0033910, 0.0047480, 0.0036921, 0.0030915, 0.0022234, 0.0022732, 0.0016449, 0.0014489, 0.00067314, 0.0004840, 0.00024277, 0.00004702])
        y_upper_y    = np.array([0.0002293, 0.00034876, 0.00049489, 0.00064819, 0.0013558, 0.0019876, 0.0020254, 0.0017527, 0.0020428, 0.0021423, 0.0026844, 0.0031016, 0.0030763, 0.0031334, 0.0036669, 0.0038131, 0.0052646, 0.0040964, 0.0034813, 0.0025892, 0.0026528, 0.0019701, 0.0017681, 0.00085149, 0.0006308, 0.00034815, 0.00008841])
        y_lower_y    = np.array([6.2559e-05, 0.00013650, 0.00023522, 0.00034921, 0.00091988, 0.0014446, 0.0014607, 0.0012255, 0.0015012, 0.0015498, 0.0020655, 0.0023973, 0.0023289, 0.0023356, 0.0028818, 0.0030044, 0.0042675, 0.0031957, 0.0027057, 0.0019102, 0.0019568, 0.0013832, 0.0012533, 0.00050150, 0.00034036, 0.00013583, 0.000019])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        
        # Adjust for cosmology
        x      = (10**log_x) * (0.70/0.681)**2
        y      = y * (0.70/0.681)**3
        y_scat = y_scat * (0.70/0.681)**3
            
        # Create dataset
        dset = f.create_dataset("data/etg/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/etg/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/etg/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        print("Successfully created: %s/GalaxyStellarMassFunction/Kelvin2014.hdf5"%obs_dir)
              

#--------------
# H1 mass func and mass fraction func
def _create_Lagos2014_H1():
    # Create and write
    with h5py.File("%s/GalaxyHIMassFunction/Lagos2014_H1.hdf5"%obs_dir, "a") as f:
        
        # Creating metadata
        grp = f.create_group("metadata")
        grp.attrs["bibcode"]      = "2014MNRAS.443.1002L"
        grp.attrs["citation"]     = "Lagos et al. (2014)"
        grp.attrs["comment"]      = "Original data from ATLAS3D. No cosmology correction needed."
        grp.attrs["name"]         = "HI mass function and mass function fraction from ATLAS3D ready to be plotted"
        grp.attrs["plot_as"]      = "points" 
        grp.attrs["redshift"]     = 0.0
        grp.attrs["redshift_lower"] = 0.0
        grp.attrs["redshift_upper"] = 0.0
        
        # Creating cosmology metadata
        grp = f.create_group("cosmology")
        grp.attrs["H0"]         = 68.1
        grp.attrs["Neff"]       = 3.046
        grp.attrs["Ob0"]        = 0.0486
        grp.attrs["Ode0"]       = 0.693922
        grp.attrs["Om0"]        = 0.306
        grp.attrs["Tcmb0"]      = 2.7255
        grp.attrs["m_nu"]       = np.array([0.0, 0.0, 0.0])
        grp.attrs["m_nu_units"] = 'eV'
        grp.attrs["name"]       = 'Abbott22'
        
        #-------------------------------
        # Creating dataset + cosmology
        
        
        #-----------
        # y-values: ATLAS3D 1/V corrected mass function
        grp = f.create_group("data/massfunc/1Vcorr/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'HI mass function 1/V corrected'
        grp = f.create_group("data/massfunc/1Vcorr/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (H1MF)'
        
        # Data
        log_x      = np.array([6.3705, 6.6187, 6.8696, 7.1201, 7.3697, 7.6227, 7.8690, 8.1228, 8.3732, 8.6220, 8.8718, 9.1232, 9.3746, 9.6232, 9.8744])
        x          = 10**log_x
        y            = 10**np.array([-1.9976, -2.3729, -2.0529, -3.1306, -3.5021, -3.5780, -3.1822, -3.1736, -3.0329, -3.0392, -3.1097, -3.4006, -3.0342, -3.8781, -3.8833])
        y_upper_y    = 10**np.array([-1.5712, -1.9505, -1.8618, -2.6953, -3.0720, -3.2750, -2.9915, -2.9947, -2.8776, -2.8750, -2.9285, -3.1580, -2.8804, -3.4530, -3.4486])
        y_lower_y    = 10**np.array([-2.4372, -2.8106, -2.2528, -3.5604, -3.9403, -3.8790, -3.3820, -3.3700, -3.2017, -3.1992, -3.2821, -3.6550, -3.2023, -4.3141, -4.3190])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        # Create dataset
        dset = f.create_dataset("data/massfunc/1Vcorr/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/massfunc/1Vcorr/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/massfunc/1Vcorr/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        #-----------
        # y-values: ATLAS3D non-1/V corrected mass FRACTION function
        grp = f.create_group("data/massfracfunc/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'HI mass fraction, not 1/V corrected, taken from HI/L_K distribution with conversions already applied'
        grp = f.create_group("data/massfracfunc/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'HI mass fraction function, not 1/V corrected, taken from HI/L_K distribution function with conversions already applied'
        
        # Data
        log_x      = np.array([-4.1236, -3.8745, -3.6268, -3.3784, -3.1290, -2.8775, -2.6269, -2.3756, -2.1283, -1.8792, -1.6309, -1.3787, -1.1305, -0.88018, -0.37902])
        x          = 10**log_x
        y            = 10**np.array([-3.8781, -3.5823, -3.2749, -3.3948, -3.5706, -3.4017, -3.1747, -3.4025, -3.1708, -3.0969, -2.9185, -3.1816, -3.5793, -3.5786, -3.8797])
        y_upper_y    = 10**np.array([-3.4462, -3.2696, -3.0628, -3.1508, -3.2674, -3.1495, -2.9833, -3.1469, -2.9889, -2.9224, -2.7776, -2.9817, -3.2694, -3.2704, -3.4409])
        y_lower_y    = 10**np.array([-4.3150, -3.8849, -3.4909, -3.6460, -3.8802, -3.6419, -3.3684, -3.6461, -3.3737, -3.2766, -3.0663, -3.3711, -3.8790, -3.8817, -4.3111])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        # Unit conversions -> given as M*/L -> * L/M* to get M*/M* | mass-to-light ratio of K_S band for ETGs, with mass-to-light ratio of 0.5 M*/L* for ETGs
        x      = x * (1/0.5)
        #y      = 
        #y_scat = 
        
        # Create dataset
        dset = f.create_dataset("data/massfracfunc/x/values", data=x)
        dset.attrs["units"]    = 'dimensionless'
        dset = f.create_dataset("data/massfracfunc/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/massfracfunc/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        print("Successfully created: %s/GalaxyHIMassFunction/Lagos2014_H1.hdf5"%obs_dir)

# H1 mass func and mass fraction func
def _create_Lagos2014_H2():
    # Create and write
    with h5py.File("%s/GalaxyH2MassFunction/Lagos2014_H2.hdf5"%obs_dir, "a") as f:
        
        # Creating metadata
        grp = f.create_group("metadata")
        grp.attrs["bibcode"]      = "2014MNRAS.443.1002L"
        grp.attrs["citation"]     = "Lagos et al. (2014)"
        grp.attrs["comment"]      = "Original data from ATLAS3D. No cosmology correction needed."
        grp.attrs["name"]         = "H2 mass function and mass function fraction from ATLAS3D ready to be plotted, divided by 1.36 to remove He"
        grp.attrs["plot_as"]      = "points" 
        grp.attrs["redshift"]     = 0.0
        grp.attrs["redshift_lower"] = 0.0
        grp.attrs["redshift_upper"] = 0.0
        
        # Creating cosmology metadata
        grp = f.create_group("cosmology")
        grp.attrs["H0"]         = 68.1
        grp.attrs["Neff"]       = 3.046
        grp.attrs["Ob0"]        = 0.0486
        grp.attrs["Ode0"]       = 0.693922
        grp.attrs["Om0"]        = 0.306
        grp.attrs["Tcmb0"]      = 2.7255
        grp.attrs["m_nu"]       = np.array([0.0, 0.0, 0.0])
        grp.attrs["m_nu_units"] = 'eV'
        grp.attrs["name"]       = 'Abbott22'
        
        #-------------------------------
        # Creating dataset + cosmology
        
        
        #-----------
        # y-values: ATLAS3D 1/V corrected mass function
        grp = f.create_group("data/massfunc/1Vcorr/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'H2 mass function 1/V corrected, divided by 1.36 for He'
        grp = f.create_group("data/massfunc/1Vcorr/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'Phi (H2MF)'
        
        # Data
        log_x      = np.array([7.0929, 7.2954, 7.4958, 7.6934, 7.8960, 8.0955, 8.2971, 8.4990, 8.6996, 8.8991, 9.0979, 9.2980])
        x          = 10**log_x
        y            = 10**np.array([-2.6063, -1.9020, -2.7252, -2.8998, -3.0271, -2.9406, -2.8971, -3.0543, -2.9409, -3.8937, -3.5951, -3.8995])
        y_upper_y    = 10**np.array([-2.1748, -1.7718, -2.4802, -2.6863, -2.8508, -2.7971, -2.7642, -2.8928, -2.8028, -3.4681, -3.2932, -3.4711])
        y_lower_y    = 10**np.array([-3.0360, -2.0460, -2.9777, -3.1211, -3.2110, -3.0813, -3.0370, -3.2202, -3.0873, -4.3301, -3.9023, -4.3271])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        # Unit conversions -> given as M*/L -> * L/M* to get M*/M* | mass-to-light ratio of K_S band for ETGs, with mass-to-light ratio of 0.5 M*/L* for ETGs, divided by 1.36 for He
        x      = x / 1.36
        
        # Create dataset
        dset = f.create_dataset("data/massfunc/1Vcorr/x/values", data=x)
        dset.attrs["units"]    = 'Msun'
        dset = f.create_dataset("data/massfunc/1Vcorr/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/massfunc/1Vcorr/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        #-----------
        # y-values: ATLAS3D 1/V uncorrected mass FRACTION function
        grp = f.create_group("data/massfracfunc/x")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'H2 mass fraction, not 1/V corrected, taken from HI/L_K distribution with conversions already applied, divided by 1.36'
        grp = f.create_group("data/massfracfunc/y")
        grp.attrs["comoving"]    = True
        grp.attrs["description"] = 'H2 mass fraction function, not 1/V corrected, taken from HI/L_K distribution function with conversions already applied'
        
        # Data
        log_x      = np.array([-3.5051, -3.3019, -3.1039, -2.9018, -2.7025, -2.5027, -2.3029, -2.1024, -1.9027, -1.7029, -1.5041, -1.2996])
        x          = 10**log_x
        y            = 10**np.array([-3.1985, -3.2931, -3.8958, -2.9894, -3.2928, -2.8905, -3.4152, -3.0510, -2.9930, -3.2941, -3.2911, -3.1944])
        y_upper_y    = 10**np.array([-2.9971, -3.0758, -3.4584, -2.8369, -3.0733, -2.7578, -3.1684, -2.8875, -2.8355, -3.0714, -3.0754, -3.0001])
        y_lower_y    = 10**np.array([-3.3898, -3.5090, -4.3300, -3.1455, -3.5111, -3.0317, -3.6673, -3.2095, -3.1377, -3.5041, -3.5089, -3.3912])
        y_scat = np.stack((y-y_lower_y, y_upper_y-y), axis=0)
        
        # Unit conversions -> mass-to-light ratio of K_S band for ETGs, with mass-to-light ratio of 0.5 M*/L* for ETGs, divided by 1.36 for He
        x      = (x * (1/0.5))/1.36        
        #y      = 
        #y_scat = 
        
        # Create dataset
        dset = f.create_dataset("data/massfracfunc/x/values", data=x)
        dset.attrs["units"]    = 'dimensionless'
        dset = f.create_dataset("data/massfracfunc/y/values", data=y)
        dset.attrs["units"]    = 'Mpc**(-3)'
        dset = f.create_dataset("data/massfracfunc/y/scatter", data=y_scat)
        dset.attrs["units"]    = 'Mpc**(-3)'
        
        
        print("Successfully created: %s/GalaxyH2MassFunction/Lagos2014_H2.hdf5"%obs_dir)


#--------------
# Read ATLAS3D results from Davis+19 from csv, export as hdf5
def _convert_Davis2019_ATLAS3D():
    # Read CSV from Tim
    
    #pd.read_csv('file.csv', sep='\s+', )
    
    # Create and write
    # divide 1.36 for H2, un-log values
    
    print('a')

      
#================================   
# Run: 

#_create_Driver2022_new()
#_create_Kelvin2014()
        
_create_Lagos2014_H1()
_create_Lagos2014_H2()
        
        
        
        
        

