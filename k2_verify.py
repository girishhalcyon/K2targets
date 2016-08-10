import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_known(csvname, colids = [0]):
    if colids[0] == 0:
        known_csv = pd.read_csv(csvname, usecols = colids, dtype = int)
    else:
        known_csv = pd.read_csv(csvname, usecols = colids)
    return np.asarray(known_csv)

def convert_sample_colors(sample):
    mag_sample = np.empty_like(sample)
    mag_sample[:,0] = sample[:,0]
    mag_sample[:,1] = sample[:,1]
    mag_sample[:,2] = sample[:,1] - sample[:,2]
    mag_sample[:,3] = sample[:,1] - sample[:,2] - sample[:,3]
    mag_sample[:,4] = sample[:,4]
    mag_sample[:,5] = sample[:,4] - sample[:,5]
    mag_sample[:,6] = sample[:,4] - sample[:,5] - sample[:,6]
    mag_sample[:,7] = sample[:,7] + sample[:,1]
    return mag_sample

if __name__ == '__main__':
    colnames = ['KepID', 'KepMag', 'Teff', 'Metallicity',
        'Mass', 'Radius', 'Distance', 'J', 'J - H', 'H - K', 'Proper Motion',
        'g', 'g - r', 'r - i', 'z - J', 'Class']
    #colnames = ['KepID', 'KepMag', 'Teff', 'Metallicity',
    #    'Mass', 'Radius', 'Distance', 'J', 'H', 'K', 'Proper Motion',
    #    'g', 'r', 'i', 'z', 'Class']
    goodcolnames = np.asarray([1,7, 8,9,11,12, 13,14], dtype = int)
    goodcolids = [1,7, 9,11, 13,15, 17, 19]
    #goodcolnames = np.asarray([1,7, 8,9,11,12, 13], dtype = int)
    #goodcolids = [1,7, 9,11, 13,15, 17]
    fuzz_factor = 100.0
    #test_name = 'k2_cool_dwarfs_observed.csv'
    test_name = 'k2_giants_observed.csv'
    #test_name = 'k2_hotter_dwarfs_observed.csv'
    if test_name == 'k2_cool_dwarfs_observed.csv':
        test_teff = read_known(test_name, colids=[2])
        test_rad = read_known(test_name, colids = [5])

    test_ids = read_known(test_name)
    impute_catalog_name = 'C0_8_impute_catalog_color.npy'
    stell_prop_name = 'C0_8_impute_catalog_color_stell_props2.npy'
    impute_flag_name = 'C0_8_impute_flags_color.npy'
    sample_name = 'C0_8_K2_sample_color.npy'
    sample_info = np.load(sample_name)
    stell_props = np.load(stell_prop_name)
    impute_catalog = np.load(impute_catalog_name)
    impute_flags = np.load(impute_flag_name)
    catalog_ids = impute_catalog[0]
    look_ids = []
    for test_id in test_ids:
        if test_id in catalog_ids:
            look_ids = np.append(look_ids, test_id)

    all_probs_name = 'C0_8_all_probs_CRF9.npy'
    #all_probs_name = 'C0_8_impute_probs3.npy'
    all_prob_errs_name = 'C0_8_all_prob_errs_CRF9.npy'
    all_probs = np.load(all_probs_name)
    all_prob_errs = np.load(all_prob_errs_name)
    #print all_probs
    look_probs = np.empty((np.shape(all_probs)[1], len(look_ids)))
    look_prob_errs = np.empty_like(look_probs)
    for i in range(0, len(look_ids)):
        look_probs[:,i] = all_probs[np.where(catalog_ids == look_ids[i])[0]]
        look_prob_errs[:,i] = all_prob_errs[np.where(catalog_ids == look_ids[i])[0]]
    #print test_ids
    #class_names = ['M-Dwarf', 'Other Dwarf', 'Giant', 'Hot Star']
    class_names = ['Cool Dwarf', 'Giant','Hot','Other Dwarf']
    look_class_names = np.asarray([class_names[np.argmax(look_probs[:,i])] for i in range(0,len(look_ids))])
    look_class_ids = np.asarray([np.argmax(look_probs[:,i]) for i in range(0,len(look_ids))])
    test_mask = np.where(look_probs[0,:] - look_prob_errs[0,:] > np.asarray([np.max(look_probs[1:,i]) for i in range(0,len(look_probs[0,:]))]))
    all_mask = np.where(all_probs[:,0] - all_prob_errs[:,0] > np.asarray([np.max(all_probs[i,1:]) for i in range(0,len(all_probs[:,0]))]))
    naive_mask = np.where(look_class_names == class_names[0])
    print look_ids[test_mask]
    #print look_class_ids[test_mask]
    #print look_class_names[test_mask]
    #print look_probs[:,test_mask]

    new_mask = np.array([np.where(catalog_ids == look_ids[i])[0][0] for i in range(0,len(look_ids))])
    inverse_mask = np.array([np.where(test_ids == look_ids[i])[0][0] for i in range(0,len(look_ids))])
    for k in range(0,len(test_mask[0])):
        print 'Kepid: ', int(stell_props[new_mask[test_mask][k],0])
        print 'Impute Flags: ', impute_flags[new_mask[test_mask][k]]
        print look_class_names[test_mask][k]
        print look_probs[:,test_mask[0][k]]
        print look_prob_errs[:,test_mask[0][k]]

        print 'Teff: ', stell_props[new_mask[test_mask][k], 1]
        print 'Radius: ', stell_props[new_mask[test_mask][k], 2]
        print 'Mass: ', stell_props[new_mask[test_mask][k], 3]
        if test_name == 'k2_cool_dwarfs_observed.csv':
            print 'CSV Teff: ', test_teff[inverse_mask][test_mask][k][0]
            print 'CSV Radius: ', test_rad[inverse_mask][test_mask][k][0]

        for j in range(0,len(goodcolids)):
            print colnames[goodcolnames[j]], impute_catalog[goodcolids[j],new_mask[test_mask][k]]

            temp_err =  np.std(sample_info[fuzz_factor*new_mask[test_mask][k]:fuzz_factor*(new_mask[test_mask][k]+1),j])
            temp_mean = np.mean(sample_info[fuzz_factor*new_mask[test_mask][k]:fuzz_factor*(new_mask[test_mask][k]+1),j])
            print 'Fuzzy ', colnames[goodcolnames[j]], temp_mean, ' +/- ', temp_err
        print '\n'
    print len(look_class_names)

    a = len(all_mask[0])
    tot = len(all_probs[:,0])
    #print a, tot - a, tot
    #print len(test_mask[0])
    #print len(naive_mask[0])
    #print look_class_names
