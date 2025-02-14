

# Assigns correct output files and the likes
def _assign_directories(file_type):
    
    # local Mac
    if file_type == '1':
        # Raw Data Directories
        colibre_base_path = '/Users/c22048063/Documents/COLIBRE/Runs/'
        #treeDir_main    = '/Users/c22048063/Desktop/'
        
        # work directories
        COLIBRE_dir       = '/Users/c22048063/Documents/COLIBRE'
        sample_dir      = COLIBRE_dir + '/samples'
        output_dir      = COLIBRE_dir + '/outputs'
        fig_dir         = COLIBRE_dir + '/figures'
        
        return COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir
        
    # serpens
    elif file_type == '2':
        # Raw data Directories serpens
        colibre_base_path = '/home/cosmos/c22048063/COLIBRE/Runs/'
        
        # work directories
        COLIBRE_dir       = '/home/cosmos/c22048063/COLIBRE'
        sample_dir      = COLIBRE_dir + '/samples'
        output_dir      = COLIBRE_dir + '/outputs'
        fig_dir         = COLIBRE_dir + '/figures'
        
        return COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir
    # cosma8 system   
    elif file_type == '3':
        # Raw data Directories cosma8
        colibre_base_path = "/cosma8/data/dp004/colibre/Runs/"
        
        # work directories
        COLIBRE_dir       = '/cosma8/data/do019/dc-bake3'
        sample_dir      = COLIBRE_dir + '/samples'
        output_dir      = COLIBRE_dir + '/outputs'
        fig_dir         = '/cosma/home/do019/dc-bake3/COLIBRE/figures'
        
        return COLIBRE_dir, colibre_base_path, sample_dir, output_dir, fig_dir
        
        
    else:
        raise Exception('nuh-uh')
    
    
    

    


    
    
    
    