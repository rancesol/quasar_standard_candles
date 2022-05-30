
import numpy as np
from numpy import sin, pi
from signal_filters import sgfilter
from get_frequencies import get_lowest_frequency
from sklearn.cluster import KMeans, MeanShift


#def curve_gen(T,start,length,ndata,amp,freq,phase) :
#    x = np.linspace(start,start+length,ndata)
#    noise = T*(2*np.random.rand(1,len(x)) - 1)
#    return np.vstack((x, amp*sin(freq*x*2*pi + phase) + noise)).T




def find_straights(signal) :
    osc, freq = get_lowest_frequency(signal)

    # find the points with the 2nd derivative going to zero
    smoothed_signal = sgfilter(signal)
    dsignal = sgfilter(np.vstack((smoothed_signal[:-1,0],
        np.diff(smoothed_signal[:,1])/np.diff(smoothed_signal[:,0]))).T)
    ddsignal =  sgfilter(np.vstack((dsignal[:-1,0],
        np.diff(dsignal[:,1])/np.diff(dsignal[:,0]))).T)

    flat_indices = np.where((ddsignal[:,1]/np.amax(ddsignal[:,1]))**2 < 5e-3)[0]
    flat_indices = flat_indices[(flat_indices > 0.1*len(signal[:,0]))
            & (flat_indices < 0.9*len(signal[:,0]))]
    flat_points = np.vstack(([smoothed_signal[i,0] for i in flat_indices],
        [smoothed_signal[i,1] for i in flat_indices])).T


    # due to noise there may be more than one point defining the center of the straight
    # merge nearby centers of straights
    mean_shift = MeanShift(bandwidth=0.001).fit(flat_points)
    centroids = mean_shift.cluster_centers_
    if osc :
        T = 1/freq
        n_possible_straights = int(0.8*np.ptp(signal[:,0])/(T/2))
    else :
        T = 2*np.ptp(signal[:,0])
        n_possible_straights = 1
    

    # the above clustering works off of separation distance between points
    # this is good to differentiate different straights
    # but this can sometimes give multiple centers for a single straight
    # kmeans uses number of clusters as criterion for clustering so gets rid of multiple centers
    if (len(centroids[:,0]) > n_possible_straights) & (flat_points[:,0].size != 0) :
        kmeans = KMeans(n_clusters = n_possible_straights)
        kmeans.fit(centroids)
        centroids = kmeans.cluster_centers_

    
    # the merging above can still sometimes merge the points of two nearby straights
    # these will cause data at the peak or trough of oscillation to be treated as a straight
    # requiring centers to be within the envelope of the data can get rid of these
    max_noise = np.amax(abs(signal[:,1] - smoothed_signal[:,1]))

    centroid_indices = [np.where((abs(smoothed_signal[:,0] - int(len(smoothed_signal[:,0])*centroids[i,0])/len(smoothed_signal[:,0])) < 0.5*np.ptp(smoothed_signal[:,0])/len(smoothed_signal[:,0])) & (abs(smoothed_signal[:,1] - centroids[i,1]) < max_noise))[0][0]  for i in range(len(centroids[:,0])) if np.where((abs(smoothed_signal[:,0] - int(len(smoothed_signal[:,0])*centroids[i,0])/len(smoothed_signal[:,0])) < 0.5*np.ptp(smoothed_signal[:,0])/len(smoothed_signal[:,0])) & (abs(smoothed_signal[:,1] - centroids[i,1]) < max_noise))[0].size > 0]

    centroids_filtered = np.vstack(([smoothed_signal[i,0] for i in centroid_indices],
        [smoothed_signal[i,1] for i in centroid_indices])).T


    # with the all good centers of straights (though maybe less than we can see by eye)
    # the data forming the straight segments is selected
    # data within just shy of 1/4 (3/16) of the period from the center of straight is considered
    straight_data_indices = [np.where((abs(signal[:,0] - centroids_filtered[i,0]) < (3/16)*T)) for i in range(len(centroids_filtered[:,0]))]
    straight_data = [np.vstack(([signal[i,0] for i in straight_data_indices[j]], [signal[i,1] for i in straight_data_indices[j]])).T for j in range(len(centroids_filtered[:,0]))]
    

    return straight_data    # returns structure straight_data[i][j,k]



#a = curve_gen(T=0.5, start=0, length=1, ndata=1000, amp=1, freq=2, phase=pi/4)
#smoothed_a = sgfilter(a)
#straight_data = find_straights(a)
#
#
#
#import matplotlib.pyplot as plt
#fig, axs = plt.subplots(1,1, figsize=(10,8), sharex=True)
#axs.plot(a[:,0], a[:,1], c='C0', zorder=1)
#axs.plot(smoothed_a[:,0], smoothed_a[:,1], c='C0', zorder=2)
#[axs.scatter(straight_data[i][:,0], straight_data[i][:,1], s=7) for i in range(len(straight_data[:]))]
#axs.set_ylabel('a')
#
#plt.show()
