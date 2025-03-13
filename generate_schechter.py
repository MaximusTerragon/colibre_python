import numpy as np

def schechter(logm, logphi, logmstar, alpha, m_lower=None):
    """
    Generate a Schechter function (in dlogm).
    """
    
    phi = ((10**logphi) * np.log(10) * 10**((logm - logmstar) * (alpha + 1)) * np.exp(-10**(logm - logmstar)))
    
    return phi
    
def parameter_at_z0(y,z0,z1=0.2,z2=1.6,z3=3.0):
    """
    Compute parameter at redshift ‘z0‘ as a function
    of the polynomial parameters ‘y‘ and the
    redshift anchor points ‘z1‘, ‘z2‘, and ‘z3‘.
    """
    
    y1, y2, y3 = y
    a = (((y3 - y1) + (y2 - y1) / (z2 - z1) * (z1 - z3)) / (z3**2 - z1**2 + (z2**2 - z1**2) / (z2 - z1) * (z1 - z3)))
    b = ((y2 - y1) - a * (z2**2 - z1**2)) / (z2 - z1)
    c = y1 - a * z1**2 - b * z1
    
    return a * z0**2 + b * z0 + c
    
def _leja_schechter(z_in=1.0):
    # Continuity model median parameters + 1-sigma uncertainties.
    pars = {'logphi1': [-2.44, -3.08, -4.14],
                'logphi1_err': [0.02, 0.03, 0.1],
                'logphi2': [-2.89, -3.29, -3.51],
                'logphi2_err': [0.04, 0.03, 0.03],
                'logmstar': [10.79,10.88,10.84],
                'logmstar_err': [0.02, 0.02, 0.04],
                'alpha1': [-0.28],
                'alpha1_err': [0.07],
                'alpha2': [-1.48],
                'alpha2_err': [0.1]}
    
    # Draw samples from posterior assuming independent Gaussian uncertainties.
    # Then convert to mass function at ‘z=z0‘.
    draws = {}
    ndraw = 1000
    z0 = z_in
    for par in ['logphi1', 'logphi2', 'logmstar', 'alpha1', 'alpha2']:
        samp = np.array([np.random.normal(median,scale=err,size=ndraw) for median, err in zip(pars[par], pars[par+'_err'])])

        if par in ['logphi1', 'logphi2', 'logmstar']:
            draws[par] = parameter_at_z0(samp,z0)
        else:
            draws[par] = samp.squeeze()
    
    # Generate Schechter functions.
    logm = np.linspace(8, 12, 30)[:, None] # log(M) grid
    phi1 = schechter(logm, draws['logphi1'], # primary component
                        draws['logmstar'], draws['alpha1'])
    phi2 = schechter(logm, draws['logphi2'], # secondary component
                        draws['logmstar'], draws['alpha2'])
    phi = phi1 + phi2 # combined mass function

    # Compute median and 1-sigma uncertainties as a function of mass.
    phi_50, phi_84, phi_16 = np.percentile(phi, [50, 84, 16], axis=1)
    
    return logm, phi, phi_50, phi_84, phi_16
            

