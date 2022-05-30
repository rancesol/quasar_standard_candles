
import numpy as np
from numpy import sin, pi
from signal_filters import sgfilter
from get_frequencies import get_lowest_frequency
from scipy.signal import argrelmax, argrelmin
import numpy.ma as ma
import numpy.polynomial.polynomial as poly

#def curve_gen(T,start,length,ndata,amp,freq,phase) :
#    x = np.linspace(start,start+length,ndata)
#    noise = T*(2*np.random.rand(1,len(x)) - 1)
#    return np.vstack((x, amp*sin(freq*x*2*pi + phase) + 10*(x-0.5)**2 + noise)).T


def smooth(signal, window_len=11, window='hanning') :
    x = signal[:,1]
    if window_len < 3 : return signal

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    w = np.ones(window_len, 'd') if (window=='flat') else eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(), s, mode='valid')
    if window_len%2 == 0 :
        return np.vstack((signal[:,0], y[int(window_len/2):-int(window_len/2)+1])).T
    else :
        return np.vstack((signal[:,0], y[int(window_len/2-1):-int(window_len/2+1)])).T



def find_straights(signal) :

    savgol = sgfilter(signal)
    convolved = smooth(savgol, window_len=int(2e-2*len(signal[:,0])), window='flat')
    osc, freq = get_lowest_frequency(convolved)
    if osc :
        period = 1/freq


    # find local max and mins in smoothed data
    # order=10 is taken arbitrarily but seems to fit expectation best given test data
    maxima = convolved[argrelmax(convolved[:,1], order=10)[0],0]
    minima = convolved[argrelmin(convolved[:,1], order=10)[0],0]
    

    # discard any max and mins near the edges
    # cutting off first and last 1% of data is taken arbitrarily but expected to handle edge effects
    # from the convolution function (may need to increase for small sample sizes)
    maxima = maxima[(maxima > 0.01*np.amax(signal[:,0])) & (maxima < 0.99*np.amax(signal[:,0]))]
    minima = minima[(minima > 0.01*np.amax(signal[:,0])) & (minima < 0.99*np.amax(signal[:,0]))]

    
    # collect the data between the max and mins
    # regions_max starts with local maxima
    # regions_min starts with local minima
    regions_max = [signal[(signal[:,0] > maxima[i]) & (signal[:,0] < minima[minima > maxima[i]][0]), :]
            for i in range(len(maxima)) if np.any(minima > maxima[i])]
    regions_min = [signal[(signal[:,0] > minima[i]) & (signal[:,0] < maxima[maxima > minima[i]][0]), :]
            for i in range(len(minima)) if np.any(maxima > minima[i])]
    
   
    # if any of the regions overlap get rid of the overlapping region that comes first
    mask_max = [np.any(np.isin(regions_max[i][:,0], regions_max[i+1][:,0]))
            for i in range(len(regions_max[:])-1)]
    mask_min = [np.any(np.isin(regions_min[i][:,0], regions_min[i+1][:,0]))
            for i in range(len(regions_min[:])-1)]
    mask_max.append(False)
    mask_min.append(False)

    regions_max = [regions_max[i] for i in range(len(regions_max[:])) if not mask_max[i]]
    regions_min = [regions_min[i] for i in range(len(regions_min[:])) if not mask_min[i]]


    # if any maxima or minima were not detected the region will get scewed
    # checking against frequency of data will reduce the larger of these unwanted regions
    # 0.55*period is taken here to account for reasonable deviations in the max/min positions
    # (with given data it is unlikely for this step to be applied)
    if osc :
        mask_max = ma.make_mask([np.ptp(regions_max[i][:,0]) > 0.55*period
            for i in range(len(regions_max[:]))], shrink=False)
        mask_min = ma.make_mask([np.ptp(regions_min[i][:,0]) > 0.55*period
            for i in range(len(regions_min[:]))], shrink=False)
    
        regions_max = [regions_max[i] for i in range(len(regions_max[:])) if not mask_max[i]]
        regions_min = [regions_min[i] for i in range(len(regions_min[:])) if not mask_min[i]]
   

    # there may be some short time scale behavior that are picked up as straights
    # we don't want these so we discard short stretches of straight data (controlled by cutoff)
    N=10
    rgns_max_padded = [np.pad(regions_max[i][:,1], (N//2, N-1-N//2), mode='edge')
            for i in range(len(regions_max[:]))]
    rgns_min_padded = [np.pad(regions_min[i][:,1], (N//2, N-1-N//2), mode='edge')
            for i in range(len(regions_min[:]))]
    rgns_max_mvng_avg = [np.convolve(rgns_max_padded[i], np.ones(N)/N, mode='valid')
            for i in range(len(regions_max[:]))]
    rgns_min_mvng_avg = [np.convolve(rgns_min_padded[i], np.ones(N)/N, mode='valid')
            for i in range(len(regions_min[:]))]

    cutoff = 0.2
    mask_max = [np.sqrt((np.ptp(regions_max[i][:,0])/np.ptp(signal[:,0]))**2 
        + (np.ptp(rgns_max_mvng_avg[i])/np.mean(rgns_max_mvng_avg[i]))**2) < cutoff
        for i in range(len(regions_max[:]))]
    mask_min = [np.sqrt((np.ptp(regions_min[i][:,0])/np.ptp(signal[:,0]))**2
        + (np.ptp(rgns_min_mvng_avg[i])/np.mean(rgns_min_mvng_avg[i]))**2) < cutoff
        for i in range(len(regions_min[:]))]
    
    regions_max = [regions_max[i] for i in range(len(regions_max[:])) if not mask_max[i]]
    regions_min = [regions_min[i] for i in range(len(regions_min[:])) if not mask_min[i]]


    ## consider the inner 80% of data in the straight region to diminish effects of max/min
    ## 80% was taken arbitrarily
    #regions_max = [regions_max[i][int(0.1*len(regions_max[i][:,0])):int(0.9*len(regions_max[i][:,0])),:]
    #        for i in range(len(regions_max[:]))]
    #regions_min = [regions_min[i][int(0.1*len(regions_min[i][:,0])):int(0.9*len(regions_min[i][:,0])),:]
    #        for i in range(len(regions_min[:]))]


    # combine the regions into one list of arrays
    regions = regions_max + regions_min


    # fit with a polynomial (order 1)
    coeffs = [poly.polyfit(regions[i][:,0], regions[i][:,1],1)
            for i in range(len(regions[:]))]
    fits = [np.vstack((regions[i][:,0], poly.polyval(regions[i][:,0], coeffs[i]))).T
            for i in range(len(regions[:]))]


    # if the chi^2/dof is too high for one of the straight fits, discard it
    # a cutoff=3 is taken arbitrarily based off initial tests
    if len(regions) > 0 :
        cutoff = 3
        chi2dof = [sum([((regions[j][i,1] - fits[j][i,1])/regions[j][i,2])**2/(len(regions[j][:,0]) - 1)
            for i in range(len(fits[j]))]) for j in range(len(regions))]
        mask = [chi2dof[i] < cutoff for i in range(len(chi2dof))]
        coeffs = [coeffs[i] for i in range(len(coeffs)) if mask[i]]
        fits = [fits[i] for i in range(len(fits)) if mask[i]]

    return coeffs, fits



#a = curve_gen(T=1, start=0, length=1, ndata=1000, amp=1, freq=4, phase=pi/4)
#coeffs, fits = find_straights(a)
#slopes = [coeffs[i][1] for i in range(len(coeffs[:]))]
#
#import matplotlib.pyplot as plt
#fig, axs = plt.subplots(2,1, figsize=(10,8))
##axs[0].scatter(a[:,0], a[:,1], s=1, c='C0', zorder=1)
#axs[0].plot(a[:,0], a[:,1], c='C0', zorder=1)
#[axs[0].plot(fits[i][:,0], fits[i][:,1], c='k') for i in range(len(fits[:]))]
#axs[0].set_ylabel('a')
#
#axs[1].hist(slopes)
#axs[1].set_xlabel('slopes')
#
#plt.show()
#
#
