import matplotlib as plt


def set_rc_params():
    # in points - start with the body text size and play around
    
    # Add font
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    plt.rcParams['font.family'] = 'Latin Modern Roman'
    #plt.rcParams["font.family"] = "DeJavu Serif"
    #plt.rcParams['font.serif'] = 'Times New Roman'
    #plt.rcParams['font.serif'] = ['Times New Roman']
    
    
    TEXT_SIZE = 7
    SMALL_SIZE = 8
    MEDIUM_SIZE = 9
    BIGGER_SIZE = 11
    
    plt.rcParams.update({'font.size': MEDIUM_SIZE})
    plt.rcParams['legend.fontsize'] = TEXT_SIZE
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['axes.labelsize'] = MEDIUM_SIZE
    plt.rcParams['xtick.labelsize'] = SMALL_SIZE
    plt.rcParams['ytick.labelsize'] = SMALL_SIZE
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['xtick.minor.size'] = 2.5
    plt.rcParams['ytick.minor.size'] = 2.5
    plt.rcParams['xtick.minor.width'] = 0.8
    plt.rcParams['ytick.minor.width'] = 0.8
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    #plt.rcParams['ytick.right'] = True
    plt.rcParams["xtick.minor.visible"] = False
    plt.rcParams["ytick.minor.visible"] = False
    #params = {'mathtext.default': 'regular'}
    #plt.rcParams.update(params)
    
set_rc_params() 

def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

"""
def set_rc_params(mult=0.5):
    
    # Add font
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    plt.rcParams['font.family'] = 'Latin Modern Roman'
    #plt.rcParams["font.family"] = "DeJavu Serif"
    #plt.rcParams['font.serif'] = 'Times New Roman'
    #plt.rcParams['font.serif'] = ['Times New Roman']
    
    plt.rcParams.update({'font.size': 20*mult})
    plt.rcParams['legend.fontsize'] = 17.5*mult
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.labelsize'] = 20*mult
    plt.rcParams['ytick.labelsize'] = 20*mult
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['xtick.minor.size'] = 5
    plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    #plt.rcParams['ytick.right'] = True
    plt.rcParams["xtick.minor.visible"] = False
    plt.rcParams["ytick.minor.visible"] = False
    #params = {'mathtext.default': 'regular'}
    #plt.rcParams.update(params)
    plt.rcParams['axes.labelsize'] = 20*mult
    
set_rc_params(0.9) 
"""



"""def graphformat(size1, size2, size3, size4, size5, width, height):
    
    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # General graph font size formatting
    plt.rc('font', size=size1)          # controls default text sizes
    
    plt.rc('figure', titlesize=size2)   # fontsize of the figure title
    
    plt.rc('axes', titlesize=size3)     # fontsize of the axes title
    plt.rc('axes', labelsize=size3)     # fontsize of the x and y labels
    
    plt.rc('xtick', labelsize=size4)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=size4)    # fontsize of the tick labels

    plt.rc('legend', fontsize=size5)    # legend fontsize

    plt.rc('figure', figsize=(width, height))  # figure size [inches]
"""