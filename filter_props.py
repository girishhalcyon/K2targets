import numpy as np
import matplotlib.pyplot as plt

def get_mast_info(csvname):
    mastcsv = np.genfromtxt(csvname, delimiter=',', skip_header =3, usecols = np.arange(0,35), names=None)
    epics = mastcsv[:,0]
    Umags = mastcsv[:,14]
    Umagerrs = mastcsv[:,15]
    Bmags = mastcsv[:,16]
    Bmagerrs =mastcsv[:,17]
    Vmags = mastcsv[:,18]
    Vmagerrs = mastcsv[:,19]
    gmags = mastcsv[:,20]
    gmagerrs = mastcsv[:,21]
    rmags = mastcsv[:,22]
    rmagerrs = mastcsv[:,23]
    imags = mastcsv[:,24]
    imagerrs = mastcsv[:,25]
    zmags = mastcsv[:,26]
    zmagerrs = mastcsv[:,27]
    Jmags = mastcsv[:,28]
    Jmagerrs = mastcsv[:,29]
    Hmags = mastcsv[:,30]
    Hmagerrs = mastcsv[:,31]
    Kmags = mastcsv[:,32]
    Kmagerrs = mastcsv[:,33]
    kepmags = mastcsv[:,34]
    gids = np.loadtxt(csvname, usecols = [7], skiprows = 3, delimiter = ',', dtype = str)
    mags = [Umags, Bmags, Vmags, gmags, rmags, imags, zmags, Jmags, Hmags, Kmags]
    magerrs = [Umagerrs, Bmagerrs, Vmagerrs, gmagerrs, rmagerrs, imagerrs, zmagerrs, Jmagerrs, Hmagerrs, Kmagerrs]
    return epics, gids, mags, magerrs, kepmags

def find_props(propname, gids):
    propids = np.loadtxt(propname, dtype=str)
    propmask = [any([propids[x] in gids[i] for x in range(0,len(propids))]) for i in range(0,len(gids))]
    propmask = np.where(np.asarray(propmask) == True)
    return propmask

def find_intersect(epics, propa, propb):
    epicsa = epics[propa]
    epicsb = epics[propb]
    intids = np.intersect1d(epicsa,epicsb)
    intmaskA = np.nonzero(np.in1d(epicsa, epicsb))
    intmaskB = np.nonzero(np.in1d(epicsb, epicsa))
    return intids, intmaskA, intmaskB

def ra_convert(ras):
    newras = np.empty((len(ras)))
    for i in range(0,len(ras)):
        ra = ras[i].split()
        newras[i] = float(ra[0]) + float(ra[1])/60.0 + float(ra[2])/3600.0
    return newras

def dec_convert(decs):
    newdecs = np.empty((len(decs)))
    for i in range(0,len(decs)):
        dec= decs[i].split()
        if float(dec[0]) < 0.0:
            newdecs[i] = float(dec[0]) - float(dec[1])/60.0 - float(dec[2])/3600.0
        else:
            newdecs[i] = float(dec[0]) + float(dec[1])/60.0 + float(dec[2])/3600.0
    return newdecs

def blank_convert(oldarr):
    newarr = np.empty((len(oldarr)))
    for i in range(0,len(oldarr)):
        if oldarr[i] == '':
            newarr[i] = np.inf
        elif oldarr[i] == None:
            newarr[i] = np.inf
        else:
            newarr[i] = float(oldarr[i])
    return newarr

def fill_info(find_epics, epics, ras, decs, pmras, pmdecs, Teffs, metals, Rads, masses, Dists, EBVs):
    info = np.empty([11,len(find_epics)])
    for i in range(0,len(find_epics)):
        find_epic = find_epics[i]
        compindex = np.where(epics == find_epic)
        if len(compindex[0]) > 1:
            compindex = compindex[0][0]
        info[0,i] = epics[compindex]
        info[1,i] = ras[compindex]
        info[2,i] = decs[compindex]
        info[3,i] = pmras[compindex]
        info[4,i] = pmdecs[compindex]
        info[5,i] = Teffs[compindex]
        info[6,i] = metals[compindex]
        info[7,i] = Rads[compindex]
        info[8,i] = masses[compindex]
        info[9,i] = Dists[compindex]
        info[10,i] = EBVs[compindex]
    return info



if __name__ == '__main__':
    epics, gids,mags, magerrs, kepmags = get_mast_info('K2C5mast.csv')
    dwarfprops = find_props('C5dwarfprops.txt', gids)
    giantprops = find_props('C5giantprops.txt', gids)
    dgepics, dgmask, gdmask = find_intersect(epics, dwarfprops, giantprops)

    allepics = np.asarray(np.load('epics.npy'), dtype = float)
    allras = np.asarray(np.load('ras.npy'),dtype = str)
    alldecs = np.asarray(np.load('decs.npy'), dtype = str)
    allras = ra_convert(allras)
    alldecs = dec_convert(alldecs)
    allpmras = np.asarray(np.load('pmras.npy'), dtype = str)
    allpmras = blank_convert(allpmras)
    allpmdecs = np.asarray(np.load('pmdecs.npy'), dtype = str)
    allpmdecs = blank_convert(allpmdecs)
    allTeffs = blank_convert(np.asarray(np.load('Teffs.npy')))
    allmetals = blank_convert(np.asarray(np.load('metals.npy')))
    allRads = blank_convert(np.asarray(np.load('Rads.npy')))
    allmasses = blank_convert(np.asarray(np.load('masses.npy')))
    allDists = blank_convert(np.asarray(np.load('Dists.npy')))
    allEBVs = blank_convert(np.asarray(np.load('EBVs.npy')))
    dwarfinfo = fill_info(epics[dwarfprops], allepics, allras, alldecs, allpmras, allpmdecs,allTeffs, allmetals, allRads, allmasses, allDists, allEBVs)
    giantinfo = fill_info(epics[giantprops], allepics, allras, alldecs, allpmras, allpmdecs,allTeffs, allmetals, allRads, allmasses, allDists, allEBVs)
    dginfo = fill_info(dgepics, allepics, allras, alldecs, allpmras, allpmdecs,allTeffs, allmetals, allRads, allmasses, allDists, allEBVs)
    #dtestmask = np.where((8.59 < dwarfinfo[1,:]) & (dwarfinfo[1,:] < 8.8) & (18.0 < dwarfinfo[2,:]) & (dwarfinfo[2,:] < 21.0))
    plt.subplot(121)
    plt.plot(giantinfo[3,:], giantinfo[4,:], '.c', alpha = 0.2, label = 'Giants')
    plt.plot(dwarfinfo[3,:], dwarfinfo[4,:], '.k', alpha = 0.2, label = 'Dwarfs')
    plt.xlabel('Proper Motion in RA [deg]')
    plt.ylabel('Proper Motion in Dec [deg]')
    #plt.plot(dginfo[3,:], dginfo[4,:], '.r')
    plt.legend(loc = 'best')
    plt.subplot(122)
    plt.plot(giantinfo[1,:], giantinfo[2,:], '.c', alpha = 0.2)
    plt.plot(dwarfinfo[1,:], dwarfinfo[2,:], '.k', alpha = 0.2)
    plt.xlabel('RA [hrs]')
    plt.ylabel('Dec [deg]')
    #plt.plot(dginfo[1,:], dginfo[2,:], '.r')
    plt.show()
