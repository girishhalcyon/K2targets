import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.vq import whiten, kmeans2
from filter_props import read_all

def make_fuzzy_column(col, colerr = [np.nan], fuzz_factor = 100):
    samples = len(col)
    newcol = np.empty((samples*fuzz_factor))
    for i in range(0,len(col)):
        sample_mean = col[i]
        if ((np.isfinite(colerr[0])) & (np.isfinite(col[0]))):
            sample_err= colerr[i]
            newcol[fuzz_factor*i:fuzz_factor*i+fuzz_factor] = np.random.normal(sample_mean, sample_err, 100)
        else:
            newcol[fuzz_factor*i:fuzz_factor*i+fuzz_factor] = np.asarray([sample_mean]*fuzz_factor)
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
        'Poly CDPP', 'gmag', 'rmag', 'imag', 'zmag']
    nclusters = np.shape(infos)[0]
    approxcentroids = [[np.mean(infos[i,j]) for j in range(0,len(colnames))] for i in range(0,nclusters)]
    approxsigma = [[np.std(infos[i,j]) for j in range(0,len(colnames))] for i in range(0,nclusters)]
    for i in range(0,nclusters):
        print ('Cluster'+ str(i))
        for j in range(0,len(colnames)):
            print colnames[j], approxcentroids[i][j], approxsigma[i][j]
        print ('N = ' + str(len(infos[i][0])))
    if len(clustertitles) == 0:
        clustertitles = [str(i) for i in range(0,nclusters)]
    make_color_color_plots(infos, clustertitles, plottitle, saveloc=saveloc, mode= mode)
    make_single_col_plots(infos, clustertitles, plottitle, colnames, saveloc=saveloc, mode=mode)


def neighbor_impute(old_kep, old_j, old_h, old_k,
    old_pm, old_g, old_r, old_i, old_z, knn = 50):

    #Replace proper motion values of 0.0 with NaNs to mark them as unknown
    replace_pm = np.where(old_pm == 0.0)
    old_pm[replace_pm] = np.nan

    #Assign default errors to each parameter
    new_pm_err = 0.003* np.ones_like(old_pm)
    new_g_err = 0.04* np.ones_like(old_g)
    new_r_err = 0.04* np.ones_like(old_r)
    new_i_err = 0.04* np.ones_like(old_i)
    new_z_err = 0.04* np.ones_like(old_z)

    #Save variances of good parameters to constants
    kep_var = np.nanvar(old_kep)
    j_var = np.nanvar(old_j)

    old_jh = old_j - old_h
    jh_var = np.nanvar(old_jh)

    old_hk = old_h - old_k
    hk_var = np.nanvar(old_hk)

    g_var = np.nanvar(old_g)

    old_gr = old_g - old_r
    gr_var = np.nanvar(old_gr)

    old_ri = old_r - old_i
    ri_var = np.nanvar(old_ri)

    old_iz = old_i - old_z
    iz_var = np.nanvar(old_iz)
    #Create array to keep track of what parameters have been imputed
    impute_flags = np.asarray(['0']*len(old_kep))


    good_params = [old_kep, old_j, old_jh, old_hk, old_g, old_gr, old_ri, old_iz]
    good_vars = [kep_var, j_var, jh_var, hk_var, g_var, gr_var, ri_var, iz_var]
    all_old= [old_pm, old_g, old_r, old_i, old_z]
    all_new =[np.copy(old_pm), np.copy(old_g), np.copy(old_r), np.copy(old_i), np.copy(old_z)]
    all_err = [new_pm_err, new_g_err, new_r_err, new_i_err, new_z_err]
    flags = ['p', 'g', 'r', 'i', 'z']

    for i in range(0,len(all_old)):
        old_param = all_old[i]
        impute_mask = np.where((np.isfinite(old_param) == False))
        good_mask = np.where(np.isfinite(old_param))
        print len(impute_mask[0]), len(good_mask[0])
        bad_means = np.empty((len(impute_mask[0])))
        bad_errs = np.empty((len(impute_mask[0])))
        for j in range(0,len(impute_mask[0])):
            impute_flags[impute_mask][j] += flags[i] #Update imputation flag
            sqdist = np.zeros((len(good_mask[0]))) #Empty array to store distances from current unknown datapoint to known datapoints
            goodcounts = np.zeros_like(sqdist) #Keeps track of how many parameters have been used to calculate distance
            for h in range(0,len(good_params)):
                impute_copy = np.asarray([good_params[h][impute_mask][j]]*len(good_mask[0]))
                temp_sqdist = ((impute_copy -
                good_params[h][good_mask])**2.0)/good_vars[h] #Variance normalized distance
                temp_sqdist_mask = np.where((temp_sqdist != 0.0) & (np.isfinite(temp_sqdist)))
                goodcounts[temp_sqdist_mask] += 1
                sqdist[temp_sqdist_mask] += temp_sqdist[temp_sqdist_mask]
            low_counts = np.where(goodcounts < 3)
            sqdist[low_counts] = sqdist[low_counts] + np.max(sqdist)
            sqdist = sqdist/goodcounts #Divide by total number of parameters used in distance
            sort_mask = np.argsort(sqdist)
            sort_var = old_param[good_mask][sort_mask] #Sort desired parameter values acccording to distances
            sort_var = sort_var[:knn]
            sqdist = sqdist[sort_mask][:knn]
             #Accept only up to knn for imputation
            median_dist = np.median(sort_var)
            dist_sigma = abs(sort_var - median_dist)/np.std(sort_var) #Remove 3-sigma outliers
            while len(np.where(dist_sigma > 3.0)[0]) > 0:
                sqdist = sqdist[np.where(dist_sigma <= 3.0)]
                sort_var = sort_var[np.where(dist_sigma <= 3.0)]
                median_dist = np.median(sort_var)
                dist_sigma = abs(sort_var - median_dist)/np.std(sort_var)
            bad_means[j] = np.average(sort_var, weights=1.0/sqdist)
            print bad_means[j], np.std(sort_var) #Assign mean of closest datapoints as imputed value
            bad_errs[j] = np.std(sort_var) #Assign standard deviation as error
        all_new[i][impute_mask] = bad_means
        all_err[i][impute_mask] = bad_errs
        print len(np.where(np.isfinite(all_new[i]) == False)[0])
        print len(np.where(np.isfinite(all_err[i]) == False)[0])
    new_pm = all_new[0]
    new_pm_err = all_err[0]
    new_g = all_new[1]
    new_g_err = all_err[1]
    new_r = all_new[2]
    new_r_err = all_err[2]
    new_i = all_new[3]
    new_i_err = all_err[3]
    new_z = all_new[4]
    new_z_err = all_err[4]


    return new_pm, new_pm_err, new_g, new_g_err, new_r, new_r_err, new_i, new_i_err, new_z, new_z_err, impute_flags
if __name__ == '__main__':
    colnames = ['KepID', 'KepMag', 'Teff', 'log g', 'Metallicity',
        'Mass', 'Radius', 'Distance', 'Jmag', 'Hmag', 'Kmag', 'Proper Motion',
        'Poly CDPP', 'gmag', 'rmag', 'imag', 'zmag']

    fname = 'ks17.csv'
    ks17 = pd.read_csv(fname, delimiter = '|')
    '''
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

    #print 'Getting umag'
    #umag = get_pm_tot(kepid, pmfile = 'umag_all.npy')
    #np.save('umag_limit', umag)
    cdpp_tot = np.load('cdpp_limit.npy')
    pm_tot = np.load('pm_limit.npy')
    glat_tot = np.load('glat_limit.npy')
    umag = np.load('umag_limit.npy')
    gmag = np.load('gmag_limit.npy')
    rmag = np.load('rmag_limit.npy')
    imag = np.load('imag_limit.npy')
    zmag = np.load('zmag_limit.npy')

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
    umag = umag[errlimitmask]
    gmag = gmag[errlimitmask]
    rmag = rmag[errlimitmask]
    imag = imag[errlimitmask]
    zmag = zmag[errlimitmask]


    pm_tot, pm_err, gmag, gmag_err, rmag, rmag_err, imag, imag_err, zmag, zmag_err, impute_flags = neighbor_impute(kepmag,
        jmag, hmag, kmag, pm_tot, gmag, rmag, imag, zmag)

    np.save('good_kepid', kepid)

    np.save('good_kepmag', kepmag)
    np.save('good_j', jmag)
    np.save('good_j_err', jmag_err)
    np.save('good_h', hmag)
    np.save('good_h_err', hmag_err)
    np.save('good_k', kmag)
    np.save('good_k_err', kmag_err)
    np.save('imp_pm', pm_tot)
    np.save('imp_pm_err', pm_err)
    np.save('imp_g', gmag)
    np.save('imp_g_err', gmag_err)
    np.save('imp_r', rmag)
    np.save('imp_r_err', rmag_err)
    np.save('imp_i', imag)
    np.save('imp_i_err', imag_err)
    np.save('imp_z', zmag)
    np.save('imp_z_err', zmag_err)


    np.save('good_polycdpp', poly_cdpp)
    np.save('good_teff', teff)
    np.save('good_logg', logg)
    np.save('good_feh', feh)
    np.save('good_mass', mass)

    np.save('imp_flags', impute_flags)


    kepid = np.load('good_kepid.npy')

    kepmag = np.load('good_kepmag.npy')
    jmag = np.load('good_j.npy')
    jmag_err = np.load('good_j_err.npy')
    hmag = np.load('good_h.npy')
    hmag_err = np.load('good_h_err.npy')
    kmag = np.load('good_k.npy')
    kmag_err = np.load('good_k_err.npy')
    pm_tot = np.load('imp_pm.npy')
    print len(np.where(np.isfinite(pm_tot))[0])
    pm_err = np.load('imp_pm_err.npy')
    gmag = np.load('imp_g.npy')
    print len(np.where(np.isfinite(gmag))[0])
    gmag_err = np.load('imp_g_err.npy')
    rmag = np.load('imp_r.npy')
    print len(np.where(np.isfinite(rmag))[0])
    rmag_err = np.load('imp_r_err.npy')
    imag = np.load('imp_i.npy')
    print len(np.where(np.isfinite(imag))[0])
    imag_err = np.load('imp_i_err.npy')
    zmag = np.load('imp_z.npy')
    print len(np.where(np.isfinite(zmag))[0])
    zmag_err = np.load('imp_z_err.npy')


    poly_cdpp = np.load('good_polycdpp.npy')
    teff = np.load('good_teff.npy')
    logg = np.load('good_logg.npy')
    feh = np.load('good_feh.npy')
    mass = np.load('good_mass.npy')

    impute_flags = np.load('imp_flags.npy')



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
    fuzz_jhcolor = fuzz_jmag - fuzz_hmag
    fuzz_jkcolor = fuzz_jmag - fuzz_kmag
    fuzz_hkcolor = fuzz_hmag - fuzz_kmag
    fuzz_pm_tot = make_fuzzy_column(pm_tot, pm_err)
    fuzz_poly_cdpp = make_fuzzy_column(poly_cdpp)
    fuzz_gmag = make_fuzzy_column(gmag, gmag_err)
    fuzz_rmag = make_fuzzy_column(rmag, rmag_err)
    fuzz_imag = make_fuzzy_column(imag, imag_err)
    fuzz_zmag = make_fuzzy_column(zmag, zmag_err)

    colorgroup = np.asarray([fuzz_teff])
    trancolor = colorgroup.T
    cleangroup = whiten(trancolor)
    nclusters = 3
    print 'Clustering by Temperature'
    centroids, indices = kmeans2(cleangroup, nclusters, iter = 500)




    infos = [[fuzz_kepid[np.where(indices == i)],
        fuzz_kepmag[np.where(indices==i)], fuzz_teff[np.where(indices==i)],
        fuzz_logg[np.where(indices==i)], fuzz_feh[np.where(indices==i)],
        fuzz_mass[np.where(indices==i)], fuzz_radius[np.where(indices==i)],
        fuzz_dist[np.where(indices==i)], fuzz_jmag[np.where(indices==i)],
        fuzz_hmag[np.where(indices==i)], fuzz_kmag[np.where(indices==i)],
        fuzz_pm_tot[np.where(indices==i)],
        fuzz_poly_cdpp[np.where(indices==i)],
        fuzz_gmag[np.where(indices==i)],
        fuzz_rmag[np.where(indices==i)],
        fuzz_imag[np.where(indices==i)],
        fuzz_zmag[np.where(indices==i)]] for i in range(0,nclusters)]

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
        fuzz_poly_cdpp[teffmask][np.where(indices==i)],
        fuzz_gmag[teffmask][np.where(indices==i)],
        fuzz_rmag[teffmask][np.where(indices==i)],
        fuzz_imag[teffmask][np.where(indices==i)],
        fuzz_zmag[teffmask][np.where(indices==i)]]
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
        fuzz_jkcolor[teffmask][loggmask]])
    trancolor = colorgroup.T
    cleangroup = whiten(trancolor)
    #cleangroup = trancolor
    nclusters = 3
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
        fuzz_poly_cdpp[teffmask][loggmask][np.where(indices==i)],
        fuzz_gmag[teffmask][loggmask][np.where(indices==i)],
        fuzz_rmag[teffmask][loggmask][np.where(indices==i)],
        fuzz_imag[teffmask][loggmask][np.where(indices==i)],
        fuzz_zmag[teffmask][loggmask][np.where(indices==i)]]
        for i in range(0,nclusters)]


    np.save('LoggJKcluster', infos)
    approxcentroids = [np.mean(infos[i], axis =1) for i in range(0,nclusters)]
    approxsigma = [np.std(infos[i], axis =1) for i in range(0,nclusters)]
    for i in range(0,nclusters):
        print ('Cluster'+ str(i))
        for j in range(0,len(colnames)):
            print colnames[j], approxcentroids[i][j], approxsigma[i][j]
        print ('N = ' + str(len(infos[i][0])))

    clustertitles = [str(i) for i in range(0,nclusters)]
    plottitle = 'Log_g J - K'
    saveloc = 'ks17plots/'
    make_color_color_plots(infos, clustertitles, plottitle, saveloc=saveloc, mode= 'SHOW')
    make_single_col_plots(infos, clustertitles, plottitle, colnames, saveloc=saveloc, mode='SHOW')

    '''
    teffname = 'TempCluster.npy'
    cdppname = 'CDPPcluster.npy'
    loggjkname = 'LoggJKcluster.npy'
    testname = cdppname
    infos = np.load(testname)

    #all_cluster_look(teffname, 'Temperature 3 Clusters', mode = 'SHOW')
    #all_cluster_look(cdppname, 'CDPP 5 Clusters', mode = 'SHOW')
    #all_cluster_look(loggjkname, 'Log_g J - K 3 Clusters', mode = 'SHOW')
    testindex = 0
    kepids = np.unique(infos[testindex][0])
    all_ids = np.asarray(ks17.kepid)
    readmask = np.empty((len(kepids)), dtype = int)
    occurences = np.empty_like(readmask)
    for i in range(0,len(kepids)):
        readindex = np.where(all_ids == kepids[i])[0]
        readmask[i] = readindex
        nkep = len(np.where(infos[testindex][0] == kepids[i])[0])
        occurences[i] = nkep

    print '70%:', len(np.where(occurences > 70)[0])
    print '75%:', len(np.where(occurences > 75)[0])
    print '80%:', len(np.where(occurences > 80)[0])
    print '85%:', len(np.where(occurences > 85)[0])
    print '90%:', len(np.where(occurences > 90)[0])
    print '95%:', len(np.where(occurences > 95)[0])
    print '100%:', len(np.where(occurences == 100)[0])
    #print len(np.where((occurences > 95) & (rads < 1.0))[0])
    #print len(np.where((occurences == 100) & (rads > 1.0) & (rads < 2.0))[0])
    #print len(np.where((occurences == 100) & (rads > 2.0) & (rads < 15.0))[0])
    #print len(np.where((occurences == 100) & (rads > 15.0))[0])
    #plt.plot(rads, occurences, '.')
    #plt.show()

    nclusters = np.shape(infos)[0]
    approxcentroids = [[np.mean(infos[i,j]) for j in range(0,len(colnames))] for i in range(0,nclusters)]
    approxsigma = [[np.std(infos[i,j]) for j in range(0,len(colnames))] for i in range(0,nclusters)]
    for i in range(0,nclusters):
        print ('Cluster'+ str(i))
        for j in range(0,len(colnames)):
            print colnames[j], approxcentroids[i][j], approxsigma[i][j]
        print ('N = ' + str(len(infos[i][0])))

    clustertitles = [str(i) for i in range(0,nclusters)]
    plottitle = 'Log_g J - K'
    saveloc = 'ks17plots/'
    #make_color_color_plots(infos, clustertitles, plottitle, saveloc=saveloc, mode= 'SHOW')
    #make_single_col_plots(infos, clustertitles, plottitle, colnames, saveloc=saveloc, mode='SHOW')
