import os
import numpy as np
import unyt as u  # package used by swiftsimio to provide physical units
import swiftgalaxy
import swiftsimio as sw
import scipy
import csv
import time
from tqdm import tqdm
from packaging.version import Version
from swiftsimio import SWIFTDataset
from swiftgalaxy import SWIFTGalaxy, SOAP
from swiftsimio.visualisation.projection import project_gas, project_pixel_grid
from swiftsimio.visualisation import (generate_smoothing_lengths,)  # if not found try `from sw.visualisation import generate_smoothing_lengths`
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from graphformat import set_rc_params
from read_dataset_directories_colibre import _assign_directories
assert Version(sw.__version__) >= Version("9.0.2")
assert Version(swiftgalaxy.__version__) >= Version("1.2.0")


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens\n     3 cosma8           ->  ")
COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir = _assign_directories(answer)
#====================================

