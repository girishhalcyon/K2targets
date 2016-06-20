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
    return epics, gids, [Umags, Bmags, Vmags, gmags rmags, imags, zmags, Jmags, Hmags, Kmags],
        [Umagerrs, Bmagerrs, Vmagerrs, gmagerrs, rmagerrs, imagerrs, zmagerrs, Jmagerrs, Hmagerrs, Kmagerrs],
        kepmags

def find_props(propname, gids):
    propids = np.loadtxt(propname, dtype=str)
    propmask = [any([propids[x] == gids[i] for x in range(0,len(propids))]) for i in range(0,len(gids))]
    propmask = np.where(np.asarray(propmask) == True)
    return propmask

def find_intersect(epics, prop1, prop2):
    if len(prop1) > len(prop2):
        propa = prop1
        propb = prop2
        print '1st array is longer, use it for indexing intersection'
    else:
        propa = prop2
        propb = prop1
        print '2nd array is longer, use it for indexing intersection'
    epicsa = epics[propa]
    epicsb = epics[propb]
    intersectids = np.intersect1d(epicsa,epicsb)
    intersectmask = np.nonzero(np.in1d(epicsa, epicsb))
    return intersectids, intersectmask

if __name__ == '__main__':
    epics, gids, mags, kepmags = get_mast_info('K2C5mastcsv')
    dwarfprops = find_props('dwarfprops.txt', gids)
    giantprops = find_props('giantprops.txt', gids)
    dgepics, dgmask = find_intersect(epics, dwarfprops, giantprops)
