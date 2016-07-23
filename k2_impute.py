import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb



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
            newarr[i] = np.nan
        else:
            newarr[i] = float(oldarr[i])
    return newarr

def read_all(colname, readindex):
    allcol = np.load(colname)
    print ('Loaded ' + colname)
    return allcol[readindex]

def pm_tot_calc(pmra, pmdec, dec, pmra_err, pmdec_err):
    pmdec = pmdec/3600.0
    pmra = pmra/3600.0
    pmra_err = pmra_err/3600.0
    pmdec_err = pmdec_err/3600.0
    pm_tot = np.sqrt(pmdec**2.0 + (pmra**2.0)*(np.cos(np.deg2rad(dec))**2.0))
    pm_err = np.sqrt(pmdec_err**2.0 + (pmra_err**2.0)*(np.cos(np.deg2rad(dec))**2.0))
    return pm_tot, pm_err

def get_C5mast_info(csvname):
    mastcsv = np.genfromtxt(csvname, delimiter=',', skip_header =3, usecols =[0], names=None)
    epics = mastcsv[:]
    gids = np.loadtxt(csvname, usecols = [7], skiprows = 3, delimiter = ',', dtype = str)
    mask = np.where(epics >= 201000000)
    return epics[mask], gids[mask]

def get_C6mast_info(csvname):
    mastcsv = np.genfromtxt(csvname, delimiter=',', skip_header =1, usecols = [0], names=None)
    epics = mastcsv[:]
    gids = np.loadtxt(csvname, usecols = [4], skiprows = 1, delimiter = ',', dtype = str)
    mask = np.where(epics >= 201000000)
    return epics[mask], gids[mask]

def get_C_other(csvname):
    mastcsv = np.genfromtxt(csvname, delimiter=',', skip_header =1, usecols = [0], names=None)
    epics = mastcsv[:]
    mask = np.where(epics >= 201000000)
    return epics[mask]

def neighbor_impute(old_kep, old_j, old_h, old_k,
    old_pm, old_pm_err,old_g, old_r, old_i, old_z,
    old_j_err, old_h_err, old_k_err, old_g_err,
    old_r_err, old_i_err, old_z_err, knn = 50):

    #Replace proper motion values of 0.0 with NaNs to mark them as unknown
    replace_pm = np.where(old_pm == 0.0)
    old_pm[replace_pm] = np.nan

    #Copy error columns
    new_pm_err = np.copy(old_pm_err)
    new_g_err = np.copy(old_g_err)
    new_r_err = np.copy(old_r_err)
    new_i_err = np.copy(old_i_err)
    new_z_err = np.copy(old_z_err)

    new_j_err = np.copy(old_j_err)
    new_h_err = np.copy(old_h_err)
    new_k_err = np.copy(old_k_err)
    #Save variances of all parameters to constants
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
    temp_flags = ['0']*len(old_kep)
    impute_flags = np.array(['0']*len(old_kep), dtype = 'S9')


    good_params = [old_kep, old_j, old_jh, old_hk, old_g, old_gr, old_ri, old_iz]
    good_vars = [kep_var, j_var, jh_var, hk_var, g_var, gr_var, ri_var,iz_var]
    all_old= [old_pm, old_g, old_r, old_i, old_z, old_j, old_h, old_k]
    all_new =[np.copy(old_pm), np.copy(old_g), np.copy(old_r), np.copy(old_i), np.copy(old_z), np.copy(old_j), np.copy(old_h), np.copy(old_k)]
    all_err = [new_pm_err, new_g_err, new_r_err, new_i_err, new_z_err, new_j_err, new_h_err, new_k_err]
    flags = ['p', 'g', 'r', 'i', 'z', 'J', 'H', 'K']



    for i in range(0,len(all_old)):
        old_param = all_old[i]
        impute_mask = np.where((np.isfinite(old_param) == False))
        good_mask = np.where(np.isfinite(old_param))
        all_good_len = len(good_mask[0])
        all_bad_len = len(impute_mask[0])
        print flags[i]
        print all_bad_len, all_good_len
        bad_means = np.empty((len(impute_mask[0])))
        bad_errs = np.empty((len(impute_mask[0])))
        for j in range(0,len(impute_mask[0])):
            print j+1, ' out of ', all_bad_len
            index = impute_mask[0][j]
            new_flag = impute_flags[index] + flags[i]
            impute_flags[index] = new_flag #Update imputation flag
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
            sqdist = sqdist[sort_mask][:knn] #Accept only up to knn for imputation
            median_dist = np.median(sort_var)
            dist_sigma = abs(sort_var - median_dist)/np.std(sort_var) #Remove 3-sigma outliers
            while len(np.where(dist_sigma > 3.0)[0]) > 0:
                sqdist = sqdist[np.where(dist_sigma <= 3.0)]
                sort_var = sort_var[np.where(dist_sigma <= 3.0)]
                median_dist = np.median(sort_var)
                dist_sigma = abs(sort_var - median_dist)/np.std(sort_var)
            bad_means[j] = np.average(sort_var, weights=1.0/sqdist)
            #print bad_means[j], np.std(sort_var) #Assign mean of closest datapoints as imputed value
            #print np.nanmean(all_new[i]), np.nanmean(all_err[i])
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
    replace_z = np.where(np.isfinite(new_z_err) == False)
    new_z_err[replace_z] = np.median(new_z_err[np.where(np.isfinite(new_z_err))])
    new_j = all_new[5]
    new_j_err = all_err[5]
    new_h = all_new[6]
    new_h_err = all_err[6]
    new_k = all_new[7]
    new_k_err = all_err[7]

    return new_pm, new_pm_err, new_g, new_g_err, new_r, new_r_err, new_i, new_i_err, new_z, new_z_err, new_j, new_j_err, new_h, new_h_err, new_k, new_k_err, impute_flags


def fill_unimpute(find_epics, epics, decs, pmras, pmdecs, Teffs, metals, Rads, masses, Dists, allmags, allmagerrs, pmra_errs, pmdec_errs):
    colnames = ['EPIC', 'KepMag', 'Teff', 'Metallicity',
        'Mass', 'Radius', 'Distance', 'Jmag', 'J_err','Hmag', 'H_err','Kmag',
        'K_err', 'gmag', 'g_err','rmag', 'r_err','imag', 'i_err','zmag',
        'z_err', 'PM', 'PM_err']


    kepmags, gmags, rmags, imags, zmags, Jmags, Hmags, Kmags = allmags
    gmag_errs, rmag_errs, imag_errs, zmag_errs, Jmag_errs, Hmag_errs, Kmag_errs = allmagerrs
    info = np.empty([len(colnames),len(find_epics)])
    compindex = np.empty((len(find_epics)), dtype = int)
    epicarray = np.load(epics)

    all_len =  len(find_epics)
    for i in range(0,len(find_epics)):
        print i+1, ' out of ', all_len
        find_epic = int(find_epics[i])
        tempindex = np.where(epicarray == find_epic)
        if len(tempindex[0]) > 1:
            compindex[i] = int(tempindex[0][0])
        else:
            compindex[i] = int(tempindex[0])


    pmRA = read_all(pmras, compindex)
    pmDEC = read_all(pmdecs, compindex)
    DEC = read_all(decs, compindex)
    pmRA_err = read_all(pmra_errs, compindex)
    pmDEC_err = read_all(pmdec_errs, compindex)
    prop_motions, pm_errs = pm_tot_calc(pmRA, pmDEC, DEC, pmRA_err, pmDEC_err)
    info[0,:] = read_all(epics, compindex)
    info[1,:] = read_all(kepmags, compindex)
    info[2,:] = read_all(Teffs, compindex)
    info[3,:] = read_all(metals, compindex)
    info[4,:] = read_all(masses, compindex)
    info[5,:] = read_all(Rads, compindex)
    info[6,:] = read_all(Dists, compindex)
    info[7,:] = read_all(Jmags, compindex)
    info[8,:] = read_all(Jmag_errs, compindex)
    info[9,:] = read_all(Hmags, compindex)
    info[10,:] = read_all(Hmag_errs, compindex)
    info[11,:] = read_all(Kmags, compindex)
    info[12,:] = read_all(Kmag_errs, compindex)
    info[13,:] = read_all(gmags, compindex)
    info[14,:] = read_all(gmag_errs, compindex)
    info[15,:] = read_all(rmags, compindex)
    info[16,:] = read_all(rmag_errs, compindex)
    info[17,:] = read_all(imags, compindex)
    info[18,:] = read_all(imag_errs, compindex)
    info[19,:] = read_all(zmags, compindex)
    info[20,:] = read_all(zmag_errs, compindex)
    info[21,:] = prop_motions
    info[22,:] = pm_errs
    return info

def impute_fill(unimpute_catalog, knn = 50):

    old_kep = unimpute_catalog[0]
    old_j = unimpute_catalog[7]
    old_h = unimpute_catalog[9]
    old_k = unimpute_catalog[11]
    old_pm = unimpute_catalog[21]
    old_pm_err = unimpute_catalog[22]
    old_g = unimpute_catalog[13]
    old_r = unimpute_catalog[15]
    old_i = unimpute_catalog[17]
    old_z = unimpute_catalog[19]
    old_j_err = unimpute_catalog[8]
    old_h_err = unimpute_catalog[10]
    old_k_err = unimpute_catalog[12]
    old_g_err = unimpute_catalog[14]
    old_r_err = unimpute_catalog[16]
    old_i_err = unimpute_catalog[18]
    old_z_err = unimpute_catalog[20]
    new_pm, new_pm_err, new_g, new_g_err, new_r, new_r_err, new_i, new_i_err, new_z, new_z_err, new_j, new_j_err, new_h, new_h_err, new_k, new_k_err, impute_flags = neighbor_impute(old_kep,
        old_j, old_h,old_k, old_pm, old_pm_err,old_g, old_r, old_i, old_z,
        old_j_err, old_h_err, old_k_err, old_g_err, old_r_err, old_i_err, old_z_err, knn = knn)

    new_catalog = np.copy(unimpute_catalog)
    new_catalog[7] = new_j
    new_catalog[8] = new_j_err
    new_catalog[9] = new_h
    new_catalog[10] = new_h_err
    new_catalog[11] = new_k
    new_catalog[12] = new_k_err
    new_catalog[13] = new_g
    new_catalog[14] = new_g_err
    new_catalog[15] = new_r
    new_catalog[16] = new_r_err
    new_catalog[17] = new_i
    new_catalog[18] = new_i_err
    new_catalog[19] = new_z
    new_catalog[20] = new_z_err
    new_catalog[21] = new_pm
    new_catalog[22] = new_pm_err

    return new_catalog, impute_flags

if __name__ == '__main__':
    ##SECTION: Generating new INFO files##
    #epics, gids,kepmags = get_C5mast_info('K2C5mast.csv')
    #epics, gids, kepmags = get_C6mast_info('K2C6mast.csv')
    #dwarfprops = find_props('C5dwarfprops.txt', gids)
    #giantprops = find_props('C6giantprops.txt', gids)
    #dgepics, dgmask, gdmask = find_intersect(epics, dwarfprops, giantprops)

    #
    '''
    allepics = 'epics.npy'
    alldecs = 'decs.npy'
    allpmras = 'pmras.npy'
    allpmdecs ='pmdecs.npy'
    allTeffs = 'Teffs.npy'
    allmetals = 'metals.npy'
    allRads ='Rads.npy'
    allmasses = 'masses.npy'
    allDists = 'Dists.npy'

    allkepmags = 'kepmags.npy'
    allgmags = 'gmags.npy'
    allrmags = 'rmags.npy'
    allimags = 'imags.npy'
    allzmags = 'zmags.npy'
    allJmags = 'Jmags.npy'
    allHmags = 'Hmags.npy'
    allKmags = 'Kmags.npy'

    allgmag_errs = 'k2_g_err.npy'
    allrmag_errs = 'k2_r_err.npy'
    allimag_errs = 'k2_i_err.npy'
    allzmag_errs = 'k2_z_err.npy'
    allJmag_errs = 'k2_J_err.npy'
    allHmag_errs = 'k2_H_err.npy'
    allKmag_errs = 'k2_K_err.npy'

    allpmRA_errs = 'k2_pmRA_err.npy'
    allpmDEC_errs = 'k2_pmDEC_err.npy'
    allmags = [allkepmags, allgmags, allrmags, allimags, allzmags, allJmags, allHmags, allKmags]
    allmagerrs = [allgmag_errs, allrmag_errs, allimag_errs, allzmag_errs, allJmag_errs, allHmag_errs, allKmag_errs]

    C0epics = get_C_other('K2C0mast.csv')
    C1epics = get_C_other('K2C1mast.csv')
    C2epics = get_C_other('K2C2mast.csv')
    C3epics = get_C_other('K2C3mast.csv')
    C4epics = get_C_other('K2C4mast.csv')
    C7epics = get_C_other('K2C7mast.csv')
    C8epics = get_C_other('K2C8mast.csv')

    C0_unimpute_catalog = fill_unimpute(C0epics, allepics, alldecs, allpmras, allpmdecs, allTeffs,
        allmetals, allRads, allmasses, allDists, allmags, allmagerrs, allpmRA_errs, allpmDEC_errs)
    np.save('C0_unimpute_catalog', C0_unimpute_catalog)

    C1_unimpute_catalog = fill_unimpute(C1epics, allepics, alldecs, allpmras, allpmdecs, allTeffs,
        allmetals, allRads, allmasses, allDists, allmags, allmagerrs, allpmRA_errs, allpmDEC_errs)
    np.save('C1_unimpute_catalog', C1_unimpute_catalog)

    C2_unimpute_catalog = fill_unimpute(C2epics, allepics, alldecs, allpmras, allpmdecs, allTeffs,
        allmetals, allRads, allmasses, allDists, allmags, allmagerrs, allpmRA_errs, allpmDEC_errs)
    np.save('C2_unimpute_catalog', C2_unimpute_catalog)

    C3_unimpute_catalog = fill_unimpute(C3epics, allepics, alldecs, allpmras, allpmdecs, allTeffs,
        allmetals, allRads, allmasses, allDists, allmags, allmagerrs, allpmRA_errs, allpmDEC_errs)
    np.save('C3_unimpute_catalog', C3_unimpute_catalog)

    C4_unimpute_catalog = fill_unimpute(C4epics, allepics, alldecs, allpmras, allpmdecs, allTeffs,
        allmetals, allRads, allmasses, allDists, allmags, allmagerrs, allpmRA_errs, allpmDEC_errs)
    np.save('C4_unimpute_catalog', C4_unimpute_catalog)

    C7_unimpute_catalog = fill_unimpute(C7epics, allepics, alldecs, allpmras, allpmdecs, allTeffs,
        allmetals, allRads, allmasses, allDists, allmags, allmagerrs, allpmRA_errs, allpmDEC_errs)
    np.save('C7_unimpute_catalog', C7_unimpute_catalog)

    C8_unimpute_catalog = fill_unimpute(C8epics, allepics, alldecs, allpmras, allpmdecs, allTeffs,
        allmetals, allRads, allmasses, allDists, allmags, allmagerrs, allpmRA_errs, allpmDEC_errs)
    np.save('C8_unimpute_catalog', C8_unimpute_catalog)

    C5_unimpute_catalog = fill_unimpute(C5epics, allepics, alldecs, allpmras, allpmdecs, allTeffs,
        allmetals, allRads, allmasses, allDists, allmags, allmagerrs, allpmRA_errs, allpmDEC_errs)
    np.save('C5_unimpute_catalog', C5_unimpute_catalog)
    C6_unimpute_catalog = fill_unimpute(C6epics, allepics, alldecs, allpmras, allpmdecs, allTeffs,
        allmetals, allRads, allmasses, allDists, allmags, allmagerrs, allpmRA_errs, allpmDEC_errs)
    np.save('C6_unimpute_catalog', C6_unimpute_catalog)
    '''
    C0_unimpute_catalog = np.load('C0_unimpute_catalog.npy')
    C1_unimpute_catalog = np.load('C1_unimpute_catalog.npy')
    C2_unimpute_catalog = np.load('C2_unimpute_catalog.npy')
    C3_unimpute_catalog = np.load('C3_unimpute_catalog.npy')
    C4_unimpute_catalog = np.load('C4_unimpute_catalog.npy')
    C5_unimpute_catalog = np.load('C5_unimpute_catalog.npy')
    C6_unimpute_catalog = np.load('C6_unimpute_catalog.npy')
    C7_unimpute_catalog = np.load('C7_unimpute_catalog.npy')
    C8_unimpute_catalog = np.load('C8_unimpute_catalog.npy')

    C0_8_unimpute_catalog = np.hstack((C0_unimpute_catalog, C1_unimpute_catalog, C2_unimpute_catalog,
        C3_unimpute_catalog, C4_unimpute_catalog,
        C5_unimpute_catalog, C6_unimpute_catalog,
        C7_unimpute_catalog, C8_unimpute_catalog))

    np.save('C0_8_unimpute_catalog', C0_8_unimpute_catalog)

    C0_8_impute_catalog, C0_8_impute_flags = impute_fill(C0_8_unimpute_catalog)
    np.save('C0_8_impute_catalog', C0_8_impute_catalog)
    np.save('C0_8_impute_flags', C0_8_impute_flags)
    print 'SAVED C0_8'
    '''
    C6_impute_catalog, C6_impute_flags = impute_fill(C6_unimpute_catalog)
    np.save('C6_impute_catalog', C6_impute_catalog)
    np.save('C6_impute_flags', C6_impute_flags)

    C5_impute_catalog = np.load('C5_impute_catalog.npy')

    C5_6_impute_catalog = np.load('C5_6_impute_catalog.npy')
    '''
    new_z_err = C0_8_impute_catalog[20]
    replace_z = np.where(np.isfinite(new_z_err) == False)
    new_z_err[replace_z] = np.median(new_z_err[np.where(np.isfinite(new_z_err))])
    C0_8_impute_catalog[20] = new_z_err


    np.save('C0_8_impute_catalog', C0_8_impute_catalog)
    for i in range(0,np.shape(C0_8_impute_catalog)[0]):
        print len(np.where(np.isfinite(C0_8_impute_catalog[i]) == False)[0])
