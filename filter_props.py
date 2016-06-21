import numpy as np
import matplotlib.pyplot as plt

def get_C5mast_info(csvname):
    mastcsv = np.genfromtxt(csvname, delimiter=',', skip_header =3, usecols = np.arange(0,35), names=None)
    epics = mastcsv[:,0]
    kepmags = mastcsv[:,34]
    gids = np.loadtxt(csvname, usecols = [7], skiprows = 3, delimiter = ',', dtype = str)
    return epics, gids, kepmags

def get_C6mast_info(csvname):
    mastcsv = np.genfromtxt(csvname, delimiter=',', skip_header =1, usecols = np.arange(0,5), names=None)
    epics = mastcsv[:,0]
    kepmags = mastcsv[:,3]
    gids = np.loadtxt(csvname, usecols = [4], skiprows = 1, delimiter = ',', dtype = str)

    return epics, gids,kepmags

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
            newarr[i] = np.nan
        elif oldarr[i] == None:
            newarr[i] = np.inf
        else:
            newarr[i] = float(oldarr[i])
    return newarr

def read_all(colname, readindex):
    allcol = np.load(colname)
    print ('Loaded ' + colname)
    return allcol[readindex]

def fill_info(find_epics, epics, ras, decs, pmras, pmdecs, Teffs, metals, Rads, masses, Dists, EBVs, allmags):
    kepmags, Bmags, Vmags, umags, gmags, rmags, imags, zmags, Jmags, Hmags, Kmags = allmags
    info = np.empty([22,len(find_epics)])
    compindex = np.empty((len(find_epics)), dtype = int)
    epicarray = np.load(epics)
    print len(find_epics)
    for i in range(0,len(find_epics)):
        print i+1
        find_epic = find_epics[i]
        tempindex = np.where(epicarray == find_epic)
        if len(tempindex[0]) > 1:
            compindex[i] = int(tempindex[0][0])
        else:
            compindex[i] = int(tempindex[0])

    info[0,:] = read_all(epics, compindex)
    info[1,:] = read_all(ras, compindex)
    info[2,:] = read_all(decs, compindex)
    info[3,:] = read_all(pmras, compindex)
    info[4,:] = read_all(pmdecs, compindex)
    info[5,:] = read_all(Teffs, compindex)
    info[6,:] = read_all(metals, compindex)
    info[7,:] = read_all(Rads, compindex)
    info[8,:] = read_all(masses, compindex)
    info[9,:] = read_all(Dists, compindex)
    info[10,:] = read_all(EBVs, compindex)
    info[11, :] = read_all(kepmags, compindex)
    info[12,:] = read_all(Bmags, compindex)
    info[13,:] = read_all(Vmags, compindex)
    info[14,:] = read_all(umags, compindex)
    info[15,:] = read_all(gmags, compindex)
    info[16,:] = read_all(rmags, compindex)
    info[17,:] = read_all(imags, compindex)
    info[18,:] = read_all(zmags, compindex)
    info[19,:] = read_all(Jmags, compindex)
    info[20,:] = read_all(Hmags, compindex)
    info[21,:] = read_all(Kmags, compindex)
    return info

def reduce_propmo_calc(Jmags, rapm, decpm, dec):
    #Following the fomula given in Section 7 of Collier-Cameron et al. 2007
    #Assuming that \mu is absolute proper motion, given by \sqrt{((\mu_\alpha)(\cos{\delta}))^2 + (\mu_delta)^2}
    decpm = decpm/3600.0 #Assuming that proper motion is given in milliarcseconds
    rapm = rapm/3600.0
    return Jmags + 5.*np.log10(np.sqrt((rapm**2.0)*(np.cos(np.deg2rad(dec))**2.0) + (decpm**2.0)))

def make_reduce_propmo_plot(infos, infotitles, infocolors, infomarkers, plottitle,saveloc = ' ', Jmags = 19, Hmags = 20, rapm = 3, decpm = 4, dec = 2, mode = 'SHOW'):
    plt.subplot(111)
    plottitle2 = plottitle + ' Reduced Proper Motion'
    plt.title(plottitle2)
    for i in range(0,len(infotitles)):
        info = infos[i]
        JHcolor = info[Jmags] - info[Hmags]
        redpropmo = reduce_propmo_calc(info[Jmags], info[rapm], info[decpm], info[dec])
        plt.scatter(JHcolor, redpropmo, c = infocolors[i], marker = infomarkers[i], label = infotitles[i])
    plt.ylim(plt.ylim()[::-1])
    plt.legend(loc='best')
    plt.xlabel('J - H')
    plt.ylabel('Reduced Proper Motion')
    if mode =='SHOW':
        plt.show()
    else:
        if mode == 'PDF':
            filesuffix = '.pdf'
        else:
            filesuffix = '.png'
        savetitle = plottitle2.replace(' ', '_')
        savein = saveloc + savetitle + filesuffix
        plt.savefig(savein)
    plt.close('all')
    print 'Done plotting Reduced Proper Motion'


def make_color_mag_plots(infos, infotitles, infocolors, infomarkers, plottitle, saveloc = '',kepmags = 11, gmags = 15, rmags = 16, imags = 17, zmags = 18, Jmags = 19, Hmags = 20, Kmags = 21, mode = 'SHOW'):
    mags = [kepmags, gmags, rmags, imags, zmags, Jmags, Hmags, Kmags]
    magtitles = ['KepMag', 'g', 'r', 'i', 'z', 'J', 'H', 'K']
    for i in range(1,len(mags) - 1):
        plt.subplot(111)
        plottitle2 = (plottitle + ' ' + magtitles[i] + ' - ' + magtitles[i+1])
        plt.title(plottitle2)
        plt.xlabel(magtitles[i] + ' - ' + magtitles[i+1])
        plt.ylabel(magtitles[0])
        for j in range(0,len(infotitles)):
            info = infos[j]
            plt.scatter(info[mags[i]] - info[mags[i+1]], info[mags[0]], c = infocolors[j], marker = infomarkers[j], label = infotitles[j])
        plt.legend(loc='best')
        if mode == 'SHOW':
            plt.show()
        else:
            if mode == 'PDF':
                filesuffix = '.pdf'
            else:
                filesuffix = '.png'
            savetitle = plottitle2.replace(' ', '_')
            savein = saveloc + savetitle + filesuffix
            plt.savefig(savein)
        plt.close('all')
    print 'Done plotting Color-Magnitude'

def make_color_color_plots(infos, infotitles, infocolors, infomarkers, plottitle, saveloc = '',kepmags = 11, gmags = 15, rmags = 16, imags = 17, zmags = 18, Jmags = 19, Hmags = 20, Kmags = 21, mode = 'SHOW'):
    mags = [kepmags, gmags, rmags, imags, zmags, Jmags, Hmags, Kmags]
    magtitles = ['KepMag', 'g', 'r', 'i', 'z', 'J', 'H', 'K']
    colortitles = [magtitles[i] + ' - ' + magtitles[i+1] for i in range(1,(len(mags) -1))]
    for i in range(0,len(colortitles)-1):
        for j in range(i+1,len(colortitles)):
            plt.subplot(111)
            plottitle2 = (plottitle + ' ' + colortitles[i] + ' vs ' + colortitles[j])
            plt.title(plottitle2)
            plt.ylabel(colortitles[i])
            plt.xlabel(colortitles[j])
            for k in range(0,len(infotitles)):
                info = infos[k]
                color1 = info[mags[i+1]] - info[mags[i + 2]]
                color2 = info[mags[j+1]] - info[mags[j+2]]
                plt.scatter(color2, color1, c = infocolors[k], marker = infomarkers[k], label=infotitles[k])
            plt.legend(loc='best')
            if mode == 'SHOW':
                plt.show()
            else:
                if mode == 'PDF':
                    filesuffix = '.pdf'
                else:
                    filesuffix = '.png'
                savetitle = plottitle2.replace(' ', '_')
                savein = saveloc + savetitle + filesuffix
                plt.savefig(savein)
            plt.close('all')
    print 'Done plotting Color-Color'




if __name__ == '__main__':
    '''
    ##SECTION: Generating new INFO files##
    #epics, gids,kepmags = get_C5mast_info('K2C5mast.csv')
    #epics, gids, kepmags = get_C6mast_info('K2C6mast.csv')
    #dwarfprops = find_props('C5dwarfprops.txt', gids)
    #giantprops = find_props('C6giantprops.txt', gids)
    #dgepics, dgmask, gdmask = find_intersect(epics, dwarfprops, giantprops)

    allepics = 'epics.npy'
    allras = 'ras.npy'
    alldecs = 'decs.npy'
    allpmras = 'pmras.npy'
    allpmdecs ='pmdecs.npy'
    allTeffs = 'Teffs.npy'
    allmetals = 'metals.npy'
    allRads ='Rads.npy'
    allmasses = 'masses.npy'
    allDists = 'Dists.npy'
    allEBVs = 'EBVs.npy'
    allkepmags = 'kepmags.npy'
    allBmags = 'Bmags.npy'
    allVmags = 'Vmags.npy'
    allumags = 'umags.npy'
    allgmags = 'gmags.npy'
    allrmags = 'rmags.npy'
    allimags = 'imags.npy'
    allzmags = 'zmags.npy'
    allJmags = 'Jmags.npy'
    allHmags = 'Hmags.npy'
    allKmags = 'Kmags.npy'
    allmags = [allkepmags, allBmags, allVmags, allumags, allgmags, allrmags, allimags, allzmags, allJmags, allHmags, allKmags]

    #dwarfinfo = fill_info(epics[dwarfprops], allepics, allras, alldecs, allpmras, allpmdecs,allTeffs, allmetals, allRads, allmasses, allDists, allEBVs, allmags)
    #np.save('C5dwarfinfo', dwarfinfo)
    giantinfo = fill_info(epics[giantprops], allepics, allras, alldecs, allpmras, allpmdecs,allTeffs, allmetals, allRads, allmasses, allDists, allEBVs, allmags)
    np.save('C6giantinfo', giantinfo)
    #dginfo = fill_info(dgepics, allepics, allras, alldecs, allpmras, allpmdecs,allTeffs, allmetals, allRads, allmasses, allDists, allEBVs, allmags)
    #np.save('C5dginfo', dginfo)
    '''
    ##SECTION: Loading pre-generated INFO files##
    dwarfinfo = np.load('C5dwarfinfo.npy')
    giantinfo = np.load('C5giantinfo.npy')
    dginfo = np.load('C5dginfo.npy')
    infos = [giantinfo, dwarfinfo, dginfo]
    infotitles = ['Giants', 'Dwarfs', 'Both?']
    infocolors = ['r', 'k', 'c']
    infomarkers = ['o', 'x', '+']
    plottitle = 'C05'
    saveloc = 'C5plots/'

    make_color_color_plots(infos, infotitles, infocolors, infomarkers,plottitle, saveloc = saveloc, mode = 'PNG')
    make_color_mag_plots(infos, infotitles, infocolors, infomarkers,plottitle, saveloc = saveloc, mode = 'PNG')
    make_reduce_propmo_plot(infos, infotitles, infocolors, infomarkers,plottitle, saveloc = saveloc, mode = 'PNG')

    dwarfinfo = np.load('C6dwarfinfo.npy')
    giantinfo = np.load('C6giantinfo.npy')
    dginfo = np.load('C6dginfo.npy')
    infos = [giantinfo, dwarfinfo, dginfo]
    infotitles = ['Giants', 'Dwarfs', 'Both?']
    infocolors = ['r', 'k', 'c']
    infomarkers = ['o', 'x', '+']
    plottitle = 'C06'
    saveloc = 'C6plots/'

    make_color_color_plots(infos, infotitles, infocolors, infomarkers,plottitle, saveloc = saveloc, mode = 'PNG')
    make_color_mag_plots(infos, infotitles, infocolors, infomarkers,plottitle, saveloc = saveloc, mode = 'PNG')
    make_reduce_propmo_plot(infos, infotitles, infocolors, infomarkers,plottitle, saveloc = saveloc, mode = 'PNG')
