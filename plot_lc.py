
import numpy as np
from data_parse import parse_data, clean_data
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit
from signal_filters import sgfilter

qlist, quasar = parse_data()
hummers = ['30.11301.499', '9.4882.332', '207.16310.1050', '58.5903.69', '13.5717.178', '17.2227.488', '53.3970.140', '61.8072.358', '69.12549.21', '68.10972.36', '9.5484.258', '64.8088.215', '22.5595.1333', '63.7365.151', '25.3469.117', '6.6572.268', '2.5873.82', '82.8403.551', '42.860.123', '9.4641.568', '207.16316.446', '206.17052.388', '5.4892.1971', '25.3712.72', '52.4565.356', '211.16765.212', '9.5239.505', '61.8199.302', '208.16034.100']
lcs = [np.vstack((quasar[qlist[n]]['time'], quasar[qlist[n]]['V'], quasar[qlist[n]]['Verr'])).T
        for n in range(len(qlist))]# if qlist[n] in hummers]
zs = [quasar[qlist[n]]['z'] for n in range(len(qlist))]# if qlist[n] in hummers]
#qlist = [qlist[n] for n in range(len(qlist)) if qlist[n] in hummers]

lcs = [clean_data(lcs[n]) for n in range(len(lcs))]
#Mavgs = [np.mean(lcs[n][:,1]) for n in range(len(lcs))]

def center_x(lc) :
    center_x = np.min(lc[:,0]) + 0.5*np.ptp(lc[:,0])
    lc[:,0] -= center_x
    return 0

[center_x(lcs[n]) for n in range(len(lcs))]
lcs = [sgfilter(lcs[n]) for n in range(len(lcs))]

def Mz_relation(z) :
    A = 8.1151956
    B = -1.20775214
    return A*np.exp(B*z)

def M_corrections(lc,z) :
    lc[:,1] = lc[:,1] * 10**(Mz_relation(z)/2.5)
    return 0

#[M_corrections(lcs[n], zs[n]) for n in range(len(lcs))]

fig, axs = plt.subplots(1,1, figsize=(10,8))
[axs.scatter(lcs[n][:,0], lcs[n][:,1], s=.6, label=qlist[n]) for n in range(len(lcs))]
#axs.set_ylim(-26,-30)
#axs.legend()
plt.show()

#def exp_fit(x, A, B, C) :
#    return A*np.exp(B*x) + C
#
#z_order = np.argsort(zs)
#fitparams = curve_fit(exp_fit, zs, Mavgs, bounds=((0,-np.inf,-np.inf),(np.inf,0,0)))
#print(fitparams)
#plt.scatter(zs, Mavgs, s=.6)
#plt.plot([zs[n] for n in z_order], [exp_fit(zs[n], fitparams[0][0], fitparams[0][1], fitparams[0][2]) for n in z_order])
#plt.show()
