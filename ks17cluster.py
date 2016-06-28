import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.vq import whiten, kmeans2
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN as dbs
from sklearn.preprocessing import StandardScaler


def make_fuzzy_column(col, colerr = [np.nan]):
    samples = len(col)
    newcol = np.empty((samples*100))
    for i in range(0,len(col)):
        sample_mean = col[i]
        if np.isfinite(colerr[0]):
            sample_err= colerr[i]
            newcol[100*i:100*i+100] = np.random.normal(sample_mean, sample_err, 100)
        else:
            newcol[100*i:100*i+100] = np.asarray([sample_mean]*100)
    return newcol

def get_pm_tot(kepids, pmfile = 'pm_all.npy'):
    pm_all = np.load(pmfile)
    all_ids = pm_all[0]
    all_pm = pm_all[1]
    pm_tot = np.empty((len(kepids)), dtype = float)
    count = 1
    for i in range(0,len(kepids)):
        findid = np.where(all_ids == kepids[i])
        if len(findid[0]) > 1:
            findid = findid[0][0]
            pm_tot[i] = float(all_pm[findid])
        elif len(findid[0]) == 0:
            pm_tot[i] = np.nan
        else:
            pm_tot[i] = float(all_pm[findid])
    return pm_tot
def make_color_mag_plots(infos, infotitles, plottitle, saveloc = '',kepmags = 1, Jmags = 8, Hmags = 9, Kmags = 10, mode = 'SHOW'):
    mags = [kepmags, Jmags, Hmags, Kmags]
    magtitles = ['KepMag', 'J', 'H', 'K']
    for i in range(1,len(mags) - 1):
        plt.subplot(111)
        plottitle2 = (plottitle + ' ' + magtitles[i] + ' - ' + magtitles[i+1])
        plt.title(plottitle2)
        plt.xlabel(magtitles[i] + ' - ' + magtitles[i+1])
        plt.ylabel(magtitles[0])
        for j in range(0,len(infotitles)):
            info = infos[j]
            plt.plot(info[mags[i]] - info[mags[i+1]], info[mags[0]], '.', label = infotitles[j], alpha = 0.2)
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

def make_single_col_plots(infos, infotitles, plottitle, colnames, saveloc = '', kepmag = 1, mode = 'SHOW'):
    info1 = infos[0]
    numcols = np.shape(info1)[0]
    for i in range(1,numcols):
        if i != kepmag:
            plt.subplot(111)
            plottitle2 = (plottitle + ' ' + colnames[i] + ' vs ' + 'KepMag')
            plt.xlabel('KepMag')
            plt.ylabel(colnames[i])
            plt.title(plottitle2)
            for j in range(0,len(infotitles)):
                info = infos[j]
                plt.plot(info[kepmag], info[i], '.', label = infotitles[j], alpha = 0.2)
            plt.legend(loc = 'best')
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
    print 'Done plotting single columns'

def make_color_color_plots(infos, infotitles, plottitle, saveloc = '',kepmags = 1, Jmags = 8, Hmags = 9, Kmags = 10, mode = 'SHOW'):
    mags = [kepmags, Jmags, Hmags, Kmags]
    magtitles = ['KepMag','J', 'H', 'K']
    colortitles = [magtitles[i] + ' - ' + magtitles[j] for i in range(1,(len(mags) -1)) for j in range(i+1, len(mags))]
    color1s = [0, 0]
    color2s = [1, 2]
    for i in range(0,2):
            plt.subplot(111)
            plottitle2 = (plottitle + ' ' + colortitles[color2s[i]] + ' vs ' + colortitles[color1s[i]])
            plt.title(plottitle2)
            plt.ylabel(colortitles[color2s[i]])
            plt.xlabel(colortitles[color1s[i]])
            for k in range(0,len(infotitles)):
                info = infos[k]
                colors = [info[Jmags] - info[Hmags], info[Jmags] - info[Kmags], info[Hmags] - info[Kmags]]
                color1 = colors[color1s[i]]
                color2 = colors[color2s[i]]
                plt.plot(color1, color2, '.', label=infotitles[k], alpha = 0.2)
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

def all_cluster_look(clustername, plottitle, saveloc = 'ks17plots/', mode = 'SHOW', clustertitles = []):
    infos = np.load(clustername)
    colnames = ['KepID', 'KepMag', 'Teff', 'log g', 'Metallicity',
        'Mass', 'Radius', 'Distance', 'Jmag', 'Hmag', 'Kmag', 'Proper Motion',
        'CDPP', 'Poly CDPP']
    nclusters = np.shape(infos)[0]
    approxcentroids = [[np.mean(infos[i,j]) for j in range(0,14)] for i in range(0,nclusters)]
    approxsigma = [[np.std(infos[i,j]) for j in range(0,14)] for i in range(0,nclusters)]
    for i in range(0,nclusters):
        print ('Cluster'+ str(i))
        for j in range(0,len(colnames)):
            print colnames[j], approxcentroids[i][j], approxsigma[i][j]
        print ('N = ' + str(len(infos[i][0])))
    if len(clustertitles) == 0:
        clustertitles = [str(i) for i in range(0,nclusters)]
    make_color_color_plots(infos, clustertitles, plottitle, saveloc=saveloc, mode= mode)
    make_single_col_plots(infos, clustertitles, plottitle, colnames, saveloc=saveloc, mode=mode)

if __name__ == '__main__':
    '''
    fname = 'ks17.csv'
    ks17 = pd.read_csv(fname, delimiter = '|')
    kepid = np.asarray(ks17.kepid)
    teff = np.asarray(ks17.teff, dtype = float)
    teff_err1 = np.asarray(ks17.teff_err1, dtype = float)
    teff_err2 = np.asarray(ks17.teff_err2, dtype = float)
    teff_err = np.max([abs(teff_err1), abs(teff_err2)], axis = 0)
    logg = np.asarray(ks17.logg)
    logg_err1 = np.asarray(ks17.logg_err1)
    logg_err2 = np.asarray(ks17.logg_err2)
    logg_err = np.max([abs(logg_err1), abs(logg_err2)], axis = 0)
    feh = np.asarray(ks17.feh)
    mass = np.asarray(ks17.mass)
    radius = np.asarray(ks17.st_radius)
    kepmag = np.asarray(ks17.kepmag)
    dist = np.asarray(ks17.dist)
    jmag = np.asarray(ks17.jmag)
    jmag_err = np.asarray(ks17.jmag_err)
    hmag = np.asarray(ks17.hmag)
    hmag_err = np.asarray(ks17.hmag_err)
    kmag = np.asarray(ks17.kmag)
    kmag_err = np.asarray(ks17.kmag_err)
    teff_prov = np.asarray(ks17.teff_prov)
    logg_prov = np.asarray(ks17.logg_prov)
    feh_prov = np.asarray(ks17.feh_prov)

    logg_prov_AST = np.asarray(['AST' in logg_prov[i] for i in range(0,len(logg_prov))])
    logg_prov_SPE = np.asarray(['SPE' in logg_prov[i] for i in range(0,len(logg_prov))])
    teff_prov_SPE = np.asarray(['SPE' in teff_prov[i] for i in range(0,len(logg_prov))])
    feh_prov_SPE = np.asarray(['SPE' in feh_prov[i] for i in range(0,len(logg_prov))])
    maskprovs = np.where(((logg_prov_AST) |(logg_prov_SPE)) & (teff_prov_SPE) & (feh_prov_SPE) & (teff < 20000.0))

    kepid = kepid[maskprovs]
    teff = teff[maskprovs]
    teff_err = teff_err[maskprovs]
    logg = logg[maskprovs]
    logg_err = logg_err[maskprovs]
    feh = feh[maskprovs]
    mass = mass[maskprovs]
    radius = radius[maskprovs]
    kepmag = kepmag[maskprovs]
    dist = dist[maskprovs]
    jmag = jmag[maskprovs]
    jmag_err = jmag_err[maskprovs]
    hmag = hmag[maskprovs]
    hmag_err = hmag_err[maskprovs]
    kmag = kmag[maskprovs]
    kmag_err = kmag_err[maskprovs]


    maskfinite = np.where((np.isfinite(kepmag)) & (np.isfinite(jmag))
        & (np.isfinite(hmag)) & (np.isfinite(kmag)))
    kepid = kepid[maskfinite]
    teff = teff[maskfinite]
    teff_err = teff_err[maskfinite]
    logg = logg[maskfinite]
    logg_err = logg_err[maskfinite]
    feh = feh[maskfinite]
    mass = mass[maskfinite]
    radius = radius[maskfinite]
    kepmag = kepmag[maskfinite]
    dist = dist[maskfinite]
    jmag = jmag[maskfinite]
    jmag_err = jmag_err[maskfinite]
    hmag = hmag[maskfinite]
    hmag_err = hmag_err[maskfinite]
    kmag = kmag[maskfinite]
    kmag_err = kmag_err[maskfinite]

    errfinitemask = np.where(np.isfinite(teff_err) & np.isfinite(jmag_err) & np.isfinite(hmag_err) & np.isfinite(kmag_err))
    kepid = kepid[errfinitemask]
    teff = teff[errfinitemask]
    teff_err = teff_err[errfinitemask]
    logg = logg[errfinitemask]
    logg_err = logg_err[errfinitemask]
    feh = feh[errfinitemask]
    mass = mass[errfinitemask]
    radius = radius[errfinitemask]
    kepmag = kepmag[errfinitemask]
    dist = dist[errfinitemask]
    jmag = jmag[errfinitemask]
    jmag_err = jmag_err[errfinitemask]
    hmag = hmag[errfinitemask]
    hmag_err = hmag_err[errfinitemask]
    kmag = kmag[errfinitemask]
    kmag_err = kmag_err[errfinitemask]
    #print 'Getting CDPP'
    #cdpp_tot = get_pm_tot(kepid, pmfile = 'cdpp_all.npy')
    #np.save('cdpp_limit', cdpp_tot)
    cdpp_tot = np.load('cdpp_limit.npy')
    pm_tot = np.load('pm_limit.npy')
    temppoly = [kepmag, kepid, jmag_err, hmag_err, kmag_err, pm_tot, cdpp_tot]

    errlimitmask = np.where((jmag_err < 8.0) & (hmag_err < 8.0)
        & (kmag_err < 8.0) & (np.isfinite(pm_tot)) & (np.isfinite(cdpp_tot)))
    kepid = kepid[errlimitmask]
    teff = teff[errlimitmask]
    teff_err = teff_err[errlimitmask]
    logg = logg[errlimitmask]
    logg_err = logg_err[errlimitmask]
    feh = feh[errlimitmask]
    mass = mass[errlimitmask]
    radius = radius[errlimitmask]
    kepmag = kepmag[errlimitmask]
    dist = dist[errlimitmask]
    jmag = jmag[errlimitmask]
    jmag_err = jmag_err[errlimitmask]
    hmag = hmag[errlimitmask]
    hmag_err = hmag_err[errlimitmask]
    kmag = kmag[errlimitmask]
    kmag_err = kmag_err[errlimitmask]
    pm_tot = pm_tot[errlimitmask]
    cdpp_tot = cdpp_tot[errlimitmask]
    magcdppfit = np.polyfit(kepmag, cdpp_tot, 5)
    poly_cdpp = cdpp_tot/np.polyval(magcdppfit, kepmag)


    jhcolor = jmag - hmag
    jhcolor_err = np.sqrt((jmag_err**2.0) + (hmag_err**2.0))
    jkcolor = jmag - kmag
    jkcolor_err = np.sqrt((jmag_err**2.0) + (kmag_err**2.0))
    hkcolor = hmag - kmag
    hkcolor_err = np.sqrt((hmag_err**2.0) + (kmag_err**2.0))

    fuzz_kepid = make_fuzzy_column(kepid)
    fuzz_kepmag = make_fuzzy_column(kepmag)
    fuzz_teff = make_fuzzy_column(teff, teff_err)
    fuzz_logg = make_fuzzy_column(logg, logg_err)
    fuzz_feh = make_fuzzy_column(feh)
    fuzz_mass = make_fuzzy_column(mass)
    fuzz_radius = make_fuzzy_column(radius)
    fuzz_dist = make_fuzzy_column(dist)
    fuzz_jmag = make_fuzzy_column(jmag, jmag_err)
    fuzz_hmag = make_fuzzy_column(hmag, hmag_err)
    fuzz_kmag = make_fuzzy_column(kmag, kmag_err)
    fuzz_jhcolor = make_fuzzy_column(jhcolor, jhcolor_err)
    fuzz_jkcolor = make_fuzzy_column(jkcolor, jkcolor_err)
    fuzz_hkcolor = make_fuzzy_column(hkcolor, hkcolor_err)
    fuzz_teff_err = make_fuzzy_column(teff_err)
    fuzz_logg_err = make_fuzzy_column(logg_err)
    fuzz_jmag_err = make_fuzzy_column(jmag_err)
    fuzz_hmag_err = make_fuzzy_column(hmag_err)
    fuzz_kmag_err = make_fuzzy_column(kmag_err)
    fuzz_pm_tot = make_fuzzy_column(pm_tot)
    fuzz_cdpp_tot = make_fuzzy_column(cdpp_tot)
    fuzz_poly_cdpp = make_fuzzy_column(poly_cdpp)

    colorgroup = np.asarray([fuzz_teff])
    trancolor = colorgroup.T
    cleangroup = whiten(trancolor)
    nclusters = 3
    print 'Clustering by Temperature'
    centroids, indices = kmeans2(cleangroup, nclusters, iter = 500)

    colnames = ['KepID', 'KepMag', 'Teff', 'log g', 'Metallicity',
        'Mass', 'Radius', 'Distance', 'Jmag', 'Hmag', 'Kmag', 'Proper Motion',
        'CDPP', 'Poly CDPP']


    infos = [[fuzz_kepid[np.where(indices == i)],
        fuzz_kepmag[np.where(indices==i)], fuzz_teff[np.where(indices==i)],
        fuzz_logg[np.where(indices==i)], fuzz_feh[np.where(indices==i)],
        fuzz_mass[np.where(indices==i)], fuzz_radius[np.where(indices==i)],
        fuzz_dist[np.where(indices==i)], fuzz_jmag[np.where(indices==i)],
        fuzz_hmag[np.where(indices==i)], fuzz_kmag[np.where(indices==i)],
        fuzz_pm_tot[np.where(indices==i)],
        fuzz_cdpp_tot[np.where(indices==i)],
        fuzz_poly_cdpp[np.where(indices==i)]] for i in range(0,nclusters)]

    np.save('TempCluster', infos)
    approxcentroids = [np.mean(infos[i], axis =1) for i in range(0,nclusters)]
    approxsigma = [np.std(infos[i], axis =1) for i in range(0,nclusters)]
    teffmin = 20000.0
    for i in range(0,nclusters):
        print ('Cluster'+ str(i))
        if approxcentroids[i][2] < teffmin:
            teffmin = approxcentroids[i][2]
            teffindex = i
        for j in range(0,len(colnames)):
            print colnames[j], approxcentroids[i][j], approxsigma[i][j]

        print ('N = ' + str(len(infos[i][0])))

    teffmask = np.where(indices == teffindex)

    colorgroup = np.asarray([fuzz_logg[teffmask],
        fuzz_poly_cdpp[teffmask]])
    trancolor = colorgroup.T
    cleangroup = whiten(trancolor)
    #cleangroup = trancolor
    nclusters = 5
    centroids, indices = kmeans2(cleangroup, nclusters, iter = 2500)

    infos = [[fuzz_kepid[teffmask][np.where(indices == i)],
        fuzz_kepmag[teffmask][np.where(indices==i)],
        fuzz_teff[teffmask][np.where(indices==i)],
        fuzz_logg[teffmask][np.where(indices==i)],
        fuzz_feh[teffmask][np.where(indices==i)],
        fuzz_mass[teffmask][np.where(indices==i)],
        fuzz_radius[teffmask][np.where(indices==i)],
        fuzz_dist[teffmask][np.where(indices==i)],
        fuzz_jmag[teffmask][np.where(indices==i)],
        fuzz_hmag[teffmask][np.where(indices==i)],
        fuzz_kmag[teffmask][np.where(indices==i)],
        fuzz_pm_tot[teffmask][np.where(indices==i)],
        fuzz_cdpp_tot[teffmask][np.where(indices==i)],
        fuzz_poly_cdpp[teffmask][np.where(indices==i)]]
        for i in range(0,nclusters)]

    np.save('CDPPcluster', infos)
    approxcentroids = [np.mean(infos[i], axis =1) for i in range(0,nclusters)]
    approxsigma = [np.std(infos[i], axis =1) for i in range(0,nclusters)]
    loggmax = 2.0
    for i in range(0,nclusters):
        print ('Cluster'+ str(i))
        if approxcentroids[i][3] > loggmax:
            loggmax = approxcentroids[i][3]
            loggindex = i
        for j in range(0,len(colnames)):
            print colnames[j], approxcentroids[i][j], approxsigma[i][j]
        print ('N = ' + str(len(infos[i][0])))
    print loggindex
    loggmask = np.where(indices == loggindex)

    colorgroup = np.asarray([fuzz_logg[teffmask][loggmask],
        fuzz_jhcolor[teffmask][loggmask]])
    trancolor = colorgroup.T
    cleangroup = whiten(trancolor)
    #cleangroup = trancolor
    nclusters = 2
    centroids, indices = kmeans2(cleangroup, nclusters, iter = 2500)

    infos = [[fuzz_kepid[teffmask][loggmask][np.where(indices == i)],
        fuzz_kepmag[teffmask][loggmask][np.where(indices==i)],
        fuzz_teff[teffmask][loggmask][np.where(indices==i)],
        fuzz_logg[teffmask][loggmask][np.where(indices==i)],
        fuzz_feh[teffmask][loggmask][np.where(indices==i)],
        fuzz_mass[teffmask][loggmask][np.where(indices==i)],
        fuzz_radius[teffmask][loggmask][np.where(indices==i)],
        fuzz_dist[teffmask][loggmask][np.where(indices==i)],
        fuzz_jmag[teffmask][loggmask][np.where(indices==i)],
        fuzz_hmag[teffmask][loggmask][np.where(indices==i)],
        fuzz_kmag[teffmask][loggmask][np.where(indices==i)],
        fuzz_pm_tot[teffmask][loggmask][np.where(indices==i)],
        fuzz_cdpp_tot[teffmask][loggmask][np.where(indices==i)],
        fuzz_poly_cdpp[teffmask][loggmask][np.where(indices==i)]]
        for i in range(0,nclusters)]


    np.save('LoggJHcluster', infos)
    approxcentroids = [np.mean(infos[i], axis =1) for i in range(0,nclusters)]
    approxsigma = [np.std(infos[i], axis =1) for i in range(0,nclusters)]
    for i in range(0,nclusters):
        print ('Cluster'+ str(i))
        for j in range(0,len(colnames)):
            print colnames[j], approxcentroids[i][j], approxsigma[i][j]
        print ('N = ' + str(len(infos[i][0])))
    '''
    teffname = 'TempCluster.npy'
    cdppname = 'CDPPcluster.npy'
    loggjhname = 'LoggJHcluster.npy'
    all_cluster_look(teffname, 'Temperature 3 Clusters', mode = 'PNG')
    all_cluster_look(cdppname, 'CDPP 5 Clusters', mode = 'PNG')
    all_cluster_look(loggjhname, 'Log_g J - H 2 Clusters', mode = 'PNG')
