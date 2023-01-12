
import os
import os.path
import glob
import numpy as np
import scipy.integrate as integrate
#from progress.bar import IncrementalBar
from tqdm import tqdm


def E(z) :      # dimensionless Hubble parameter
    OmR = 9e-5
    OmM = 0.321
    OmL = 0.679
    return np.sqrt(OmR*(1+z)**4 + OmM*(1+z)**3 + OmL)

def d_lum(z) :  # luminosity distance in pc
    h = 0.7
    dH = 3e9/h  # in pc
    return (1+z)*dH*integrate.quad(lambda x: 1/E(x), 0, z)[0]

def K(z) :
    alpha = -0.5
    return -2.5*(1+alpha)*np.log10(1+z)

def M(m,z) :
    return m - 5*(np.log10(d_lum(z)) - 1) - K(z)



def parse_data_macho(RF) :
    
    cwd = os.getcwd()
    qlist = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(cwd + '/data/*.dat')]


    with open(cwd + '/data/quasar_info.dat') as metafile :
        _ = metafile.readline()
        name, Vmn, color, Mag, redshift = zip(*[(line.split()[0], float(line.split()[3]),
            float(line.split()[4]), float(line.split()[6]), float(line.split()[5])) for line in metafile])
    
    quasar = {}
    #bar = IncrementalBar('Parsing Data', max=len(qlist)-1)
    for filename in tqdm(qlist) :
        if filename != 'quasar_info' :
            with open(cwd + '/data/' + filename + '.dat') as datafile :
                qname = str(datafile.readline().rstrip('\n'))
                data = np.loadtxt(datafile, skiprows=1)
                c = color[np.where(np.asarray(name) == qname)[0][0]]
                Mavg = Mag[np.where(np.asarray(name) == qname)[0][0]]
                z = redshift[np.where(np.asarray(name) == qname)[0][0]]
                Vmean = Vmn[np.where(np.asarray(name) == qname)[0][0]]

                if RF == 'qso' :
                    data[:,0] /= (1+z)
                    V = np.asarray([M(data[i,1],z) for i in range(len(data))])
                    R = np.asarray([M(data[i,3],z) for i in range(len(data))])
                else :
                    V = data[:,1]
                    R = data[:,3]
                Verr = data[:,2]
                Rerr = data[:,4]
                Vflux = 10**(-V/2.5)
                Vfluxerr = Verr/2.5*np.log(10)*10**(-V/2.5)
                #Rflux = 10**(-R/2.5)
                #Rfluxerr = Rerr/2.5*np.log(10)*10**(-R/2.5)
    
                quasar[qname] = {'time':{},'V':{},'Vmean':{},'Verr':{},'R':{},'Rerr':{},
                        'flux':{},'fluxerr':{},'color':{},'Mag':{},'z':{}}
                quasar[qname]['time']       = data[:,0]
                quasar[qname]['V']          = V
                quasar[qname]['Vmean']      = Vmean
                quasar[qname]['Verr']       = Verr 
                quasar[qname]['R']          = R
                quasar[qname]['Rerr']       = Rerr
                quasar[qname]['flux']       = Vflux
                quasar[qname]['fluxerr']    = Vfluxerr
                quasar[qname]['color']      = c
                quasar[qname]['Mag']        = Mavg
                quasar[qname]['z']          = z

                #bar.next()
        else :
            continue
    #bar.finish()        

    qlist.remove('quasar_info')

    return qlist, quasar


#/////////////////////////////////////////A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V/////////////////////////////////////////

def parse_data_kepler(RF) :
    
    cwd = os.getcwd()
    qlist = [os.path.splitext(os.path.basename(x))[0].lstrip('lcwerrors_')
            for x in glob.glob(cwd + '/data/kepler/*.dat')]

    Ephoton = 6.67607e-34*3e8/6600e-10  # bandpass filter = 4200 - 9000 Angstrom
    conversion = Ephoton*1e7            # 1 Joule = 1e7 ergs
    
    with open(cwd + '/data/kepler/quasar_info.dat') as metafile :
        _ = metafile.readline()
        name, redshift, Kmag = zip(*[(line.split()[0], float(line.split()[3]), float(line.split()[4]))
            for line in metafile])

    quasar = {}
    bad_qso = []
    #bar = IncrementalBar('Parsing Data', max=len(qlist)-1)
    for filename in tqdm(qlist) :
        if filename != 'quasar_info' :
            with open(cwd + '/data/kepler/lcwerrors_' + filename + '.dat') as datafile :
                qname = filename
                data = np.loadtxt(datafile, skiprows=0)
                z = redshift[np.where(np.asarray(name) == qname)[0][0]]
                Kp = Kmag[np.where(np.asarray(name) == qname)[0][0]]

                mag_correction = Kp - np.mean(-2.5*np.log10(data[:,1]*conversion))
                
                if z < 0 :
                    bad_qso.append(filename)
                    bar.next()
                else :
                    if RF == 'qso' :
                        flux = np.asarray([data[i,1]*conversion*
                            10**(2*(np.log10(d_lum(z))-1)+K(z))*10**(-mag_correction/2.5)
                            for i in range(len(data))])
                        fluxerr = np.mean(flux)/np.mean(data[:,1])*data[:,2]
                    else :
                        data[:,0] *= (1+z)  # Kepler data is given in QSO's frame
                        flux = data[:,1]*conversion*10**(-mag_correction/2.5)
                        fluxerr = data[:,2]*conversion*10**(-mag_correction/2.5)
        
                    quasar[qname] = {'time':{},'flux':{},'fluxerr':{},'z':{}}
                    quasar[qname]['time']       = data[:,0]
                    quasar[qname]['flux']       = flux
                    quasar[qname]['fluxerr']    = fluxerr
                    quasar[qname]['z']          = z
    
                    #bar.next()
        else :
            continue
    #bar.finish()        

    qlist.remove('quasar_info')
    [qlist.remove(qso) for qso in bad_qso]

    return qlist, quasar



#/////////////////////////////////////////A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V/////////////////////////////////////////



def parse_data_cosmograil(RF) :
    cwd = os.getcwd()
    qlist = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(cwd + '/data/cosmograil/*.dat')]

    with open(cwd + '/data/cosmograil/quasar_info.dat') as metafile :
        _ = metafile.readline()
        name, nimages, redshift, mags = zip(*[(
            line.split()[0],
            int(line.split()[1]),
            float(line.split()[2]),
            np.asarray(line.split()[4:], dtype=float))
            for line in metafile])


    quasar = {}
    bad_qso = []
    sets = []
    #bar = IncrementalBar('Parsing Data', max=len(qlist)-1)
    for filename in tqdm(qlist) :
        if filename != 'quasar_info' :
            with open(cwd + '/data/cosmograil/' + filename + '.dat') as datafile :
                qname = str(datafile.readline().rstrip('\n'))
                data = np.genfromtxt(datafile, skip_header=1)
                n = nimages[np.where(np.asarray(name) == qname)[0][0]]
                z = redshift[np.where(np.asarray(name) == qname)[0][0]]
                m = mags[np.where(np.asarray(name) == qname)[0][0]]
                
                if m[0]<0 :
                    bad_qso.append(filename)
                    bar.next()
                else :
                    for i in range(n) :
                        qimg = qname + '.' + str(i)
                        mag_correction = m[i] - np.mean(data[:,1+2*i])

                        if RF == 'qso' :
                            data[:,0] /= (1+z)
                            R = np.asarray([M(data[j,1+2*i]+mag_correction,z) for j in range(len(data))])
                        else :
                            R = data[:,1+2*i] + mag_correction
                        Rerr = data[:,2+2*i]
                        flux = 10**(-R/2.5)
                        fluxerr = Rerr/2.5*np.log(10)*10**(-R/2.5)

                        quasar[qimg] = {'time':{},'flux':{},'fluxerr':{},'z':{}}
                        quasar[qimg]['time']    = data[:,0]
                        quasar[qimg]['flux']    = flux
                        quasar[qimg]['fluxerr'] = fluxerr
                        quasar[qimg]['z']       = z
                        sets.append(qimg)
                        
                        #bar.next()
        else :
            continue
    #bar.finish()

    [qlist.remove(qso) for qso in bad_qso]
    return sets, quasar


#/////////////////////////////////////////A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V/////////////////////////////////////////



def parse_data_sdss(RF) :

    cwd = os.getcwd()
    qlist = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(cwd + '/data/QSO_S82/*')]


    with open(cwd + '/data/QSO_S82/DB_QSO_S82') as metafile :
        _ = metafile.readline()
        _ = metafile.readline()
        dbID, Mag, redshift, Mbh = zip(*[(line.split()[0], float(line.split()[4]), float(line.split()[6]), float(line.split()[7]))
            for line in metafile])

    quasar = {}
    #bar = IncrementalBar('Parsing Data', max=len(qlist)-1)
    for filename in qlist :
        if filename != 'DB_QSO_S82' :
            with open(cwd + '/data/QSO_S82/' + filename) as datafile :
                qname = filename
                data = np.loadtxt(datafile, skiprows=0)
                Mavg = Mag[np.where(np.asarray(dbID) == qname)[0][0]]
                z = redshift[np.where(np.asarray(dbID) == qname)[0][0]]
                Mass = Mbh[np.where(np.asarray(dbID) == qname)[0][0]]

                mask = [(magi > -99.99) for magi in data[:,7]]
                t = np.asarray([data[i,9] for i in range(len(data)) if mask[i]])
                if RF == 'qso' :
                    t /= (1+z)
                    I = np.asarray([M(data[i,10],z) for i in range(len(data)) if mask[i]])
                else :
                    I = np.asarray([data[i,10] for i in range(len(data)) if mask[i]])
                Ierr = np.asarray([data[i,11] for i in range(len(data)) if mask[i]])

                Iflux = 10**(-I/2.5)#*(1+z)
                Ifluxerr = Ierr/2.5*np.log(10)*10**(-I/2.5)#*(1+z)

                quasar[qname] = {'time':{},'I':{},'Ierr':{},'flux':{},'fuxerr':{},'Mag':{},'z':{},'Mbh':{}}
                quasar[qname]['time']       = t
                quasar[qname]['I']          = I
                quasar[qname]['Ierr']       = Ierr
                quasar[qname]['flux']       = Iflux
                quasar[qname]['fluxerr']    = Ifluxerr
                quasar[qname]['Mag']        = Mavg
                quasar[qname]['z']          = z
                quasar[qname]['Mbh']        = Mass

                #bar.next()
        else :
            continue
    #bar.finish()
                        
    qlist.remove('DB_QSO_S82')

    return qlist, quasar


#/////////////////////////////////////////A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V/////////////////////////////////////////


def clean_data(flux) :
    ## Get rid of any -99 magnitude data wich MACHO has used to mark faulty observations.
    mask = [(fi <= 1e15) and (fi > -99) for fi in flux[:,1]]
    flux = flux[mask,...]

    ## Some of the data is far off the general behavior which we assume to be due to 
    ## effects local to the observer.
    ## We drop these data points by discarding anything deviating by more than twice the
    ## average deviation in a small window.
    N = 50
    flux_padded = np.pad(flux[:,1], (N//2, N-1-N//2), mode='edge')
    moving_avg  = np.convolve(flux_padded, np.ones(N)/N, mode='valid')
    mvg_avg_padded = np.pad(moving_avg, (N//2, N-1-N//2), mode='edge')
    deviation = abs(flux[:,1] - moving_avg)

    mean = np.mean(deviation)
    mask = [(dev < 2*mean) for dev in deviation]
    flux = flux[mask,...]
    
    ## Some of the data has error a few orders of magnitude greater than the rest.
    ## So if the error in the flux is greater than the flux measurement we drop it.
    ## If we keep these data it should not effect the results since the fits are weighted
    ## by the error.
    ## We drop them mainly for plotting purposes.
    mask = [(flux[i,2] < abs(flux[i,1])) for i in range(len(flux[:,2]))]
    flux = flux[mask,...]

    return flux
