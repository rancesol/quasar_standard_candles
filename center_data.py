

import numpy as np
from numpy import sin, pi
import numpy.polynomial.polynomial as poly

#def curve_gen(T,start,length,ndata,amp,freq,phase,slope) :
#    x = np.linspace(start, start+length, ndata)
#    noise = T*(2*np.random.rand(1,len(x)) - 1)
#    return np.vstack((x, amp*sin(freq*x*2*pi + phase) + noise + slope*x)).T
#
#a = curve_gen(T=1, start=5, length=2, ndata=1000, amp=2, freq=0.4, phase=pi/4, slope=-3)


def center_data(signal) :
    
    coeffs = poly.polyfit(signal[:,0], signal[:,1], 1)
    fits = np.vstack((signal[:,0], poly.polyval(signal[:,0], coeffs))).T

    centered = np.copy(signal)

    center_x = np.max(fits[:,0]) - 0.5*np.ptp(fits[:,0])    #np.mean(fits[:,0])
    center_y = np.max(fits[:,1]) - 0.5*np.ptp(fits[:,1])

    fits[:,0] -= center_x
    fits[:,1] -= center_y

    centered[:,0] -= center_x
    centered[:,1] -= center_y

    #if coeffs[1] < 0 :
    #    fits[:,1] *= -1
    #    centered[:,1] *= -1

    return centered



#centered = center_data(a)
#
#import matplotlib.pyplot as plt
#fig, axs = plt.subplots(1,1, figsize=(10,8))
#axs.scatter(a[:,0], a[:,1], s=5, c='C0')
##axs.scatter(a_cent[:,0], a_cent[:,1], s=5, c='C0')
##axs.plot(fits[:,0], fits[:,1], c='k')
#axs.scatter(centered[:,0], centered[:,1], s=5, c='C0')
#plt.tight_layout()
#plt.show()
