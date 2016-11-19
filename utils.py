import bisect
from pylab import *
from scipy.special import erf

# Constants
eps = 1.0e-10

def convert_LIF2SRM0():
    """ Refer Chapter 9.4 of Wulfram's book 2.
    Valid for subthreshold input.
    """
    pass

def calcLIFRate(tau_m,tau_ref,h0,v_reset,theta_reset,sigma):
    """
    returns rate of an LIF receiving Gaussian white noise
    with mean h0 and std sigma,
    using eqn (13.30) of Wulfram's book.
    For perfect IF (PIF), use:
    A0 = (((theta_reset-v_r)/h0)tau_m + tau_ref)^-1
    """
    integrange = linspace((v_reset-h0)/sigma,\
                        (theta_reset-h0)/sigma,1000)
    drange = integrange[1]-integrange[0]
    return 1./( tau_m*sqrt(pi)*trapz(
                                exp(integrange**2)*(1+erf(integrange)),
                            dx=drange)
                + tau_ref )

# ###########################################
# Analysis functions
# ###########################################

def KroneckerDelta(i,j):
    if i==j: return 1.0
    else: return 0.0

def inv_f_trans_on_vector(xilist,fhatlist):
    """ Adapted from code accompanying Trousdale et al Plos CB 2012
    pass two vector arguments:  xilist, list of temporal frequencies, and fhatlist,
    list of f. coefficients at those frequencies
    ** FREQUENCIES MUST BE EVENLY SPACED, from (-N/2)/duration to ((N-1)/2)/duration **

    returns two vectors: tlist, list of times evenly spaced at dt, from [-a to a]
    and flist, list of function values at those times
    """
    N=len(xilist)
    klist = array(range(N))
    jlist = array(range(N))

    one_over_duration = xilist[2]-xilist[1]
    duration = 1.0/one_over_duration
    dt = duration/N

    tlist = -duration/2 + dt*array(range(N))

    gtildelist = fhatlist*exp(-pi*1j*jlist) # element-wise *
    ftildelist = ifft(gtildelist)

    flist = 1/dt * exp(pi*1j*N/2) * exp(-pi*1j*klist) * ftildelist # element-wise *

    return (tlist,flist)

def rate_from_spiketrains(spiketimes,spikei,fulltime,dt,nrnidxs,sigma=25e-3):
    """
    Returns a rate series of spiketimes convolved with a Gaussian kernel
     for a subset of neurons given as nrnidxs array.
    All times must be in SI units;
     remember to divide fulltime and dt by second if passing in brian params.
    """
    # normalized Gaussian kernel, integral with dt is normed to 1
    # to count as 1 spike smeared over a finite interval
    norm_factor = 1./(sqrt(2.*pi)*sigma)
    gauss_kernel = array([norm_factor*exp(-x**2/(2.*sigma**2))\
        for x in arange(-5.*sigma,5.*sigma+dt,dt)])

    kernel_len = len(gauss_kernel)
    nrnidxs_sorted = sorted(nrnidxs)
    Nidxs = len(nrnidxs_sorted)

    # need to accommodate half kernel_len on either side of fulltime
    rate_full = zeros(int(fulltime/dt))
    rate_full_conv = zeros(int(fulltime/dt)+kernel_len)
    for sidx,spikeidx in enumerate(spikei):
        # using bisect with sorted array should be faster than
        #  `if idx in nrnidxs`
        idx = bisect.bisect_left(nrnidxs_sorted,spikeidx)
        if idx!=Nidxs and spikeidx == nrnidxs_sorted[idx]:
            rate_full[int(spiketimes[sidx]/dt)] += 1
    for idx,val in enumerate(rate_full):
        rate_full_conv[idx:idx+kernel_len] += val*gauss_kernel
    # This is already in Hz,
    # since should have multiplied by dt for above convolution
    # and divided by dt to get a rate, so effectively not doing either.
    # gaussian kernel is two-tailed, so have to remove either side
    return rate_full_conv[kernel_len/2:int(fulltime/dt)+kernel_len/2]

def rate_from_spiketrain(spiketimes,spikei,fulltime,\
                            dt=1e-4,nrnidx=None,sigma=25e-3):
    """
    Returns a rate series of spiketimes convolved with a Gaussian kernel
     for a single neuron given by nrnidx.
    All times must be in SI units,
     remember to divide fulltime and dt by second if sending from Brian.
    """
    # normalized Gaussian kernel, integral with dt is normed to 1
    # to count as 1 spike smeared over a finite interval
    norm_factor = 1./(sqrt(2.*pi)*sigma)
    gauss_kernel = array([norm_factor*exp(-x**2/(2.*sigma**2))\
        for x in arange(-5.*sigma,5.*sigma+dt,dt)])

    if nrnidx is not None:
        # take spiketimes of only neuron index nrnidx
        spiketimes = spiketimes[where(spikei==nrnidx)]

    kernel_len = len(gauss_kernel)
    # need to accommodate half kernel_len on either side of fulltime
    rate_full = zeros(int(fulltime/dt)+kernel_len)
    for spiketime in spiketimes:
        idx = int(spiketime/dt)
        rate_full[idx:idx+kernel_len] += gauss_kernel
    # only the middle fulltime part of the rate series
    # This is already in Hz,
    # since should have multiplied by dt for above convolution
    # and divided by dt to get a rate, so effectively not doing either.
    return rate_full[kernel_len/2:kernel_len/2+int(fulltime/dt)]

def CV_spiketrains(spiket,spikei,tinit,tend,nrnidxs):
    """ calculate CV of excitatory neurons
     a la Lim and Goldman 2013
    """
    CV = []
    for j in nrnidxs:
        indices = where(spikei == j)
        # at least 5 spikes in this neuron to get CV
        if len(indices[0])>=5:
            spiketimes = spiket[indices]
            spiketimes = spiketimes[\
                (spiketimes>tinit) & (spiketimes<=tend)]
            # at least 5 spikes in time range to get CV
            if len(spiketimes)>=5:
                ISI = diff(spiketimes)
                CV.append( std(ISI)/mean(ISI) )
    return array(CV)
