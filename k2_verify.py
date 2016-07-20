import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_known(csvname, colids = [0]):
    known_csv = pd.read_csv(csvname, usecols = colids, dtype = int)
    return np.asarray(known_csv)



if __name__ == '__main__':
    colnames = ['KepID', 'KepMag', 'Teff', 'Metallicity',
        'Mass', 'Radius', 'Distance', 'Jmag', 'Hmag', 'Kmag', 'Proper Motion',
        'gmag', 'rmag', 'imag', 'zmag', 'Class']
    goodcolnames = np.asarray([1,7, 8,9,11,12, 13,14], dtype = int)
    goodcolids = [1,7, 9,11, 13,15, 17, 19]
    test_name = 'k2_cool_dwarfs_observed.csv'
    #test_name = 'k2_giants_observed.csv'
    #test_name = 'k2_hotter_dwarfs_observed.csv'
    test_ids = read_known(test_name)
    impute_catalog_name = 'C5_impute_catalog.npy'
    impute_flag_name = 'C5_impute_flags.npy'
    impute_catalog = np.load(impute_catalog_name)
    impute_flags = np.load(impute_flag_name)
    catalog_ids = impute_catalog[0]
    look_ids = []
    for test_id in test_ids:
        if test_id in catalog_ids:
            look_ids = np.append(look_ids, test_id)

    all_probs_name = 'C5_all_probs_CRF2.npy'
    all_probs = np.load(all_probs_name)
    look_probs = np.empty((np.shape(all_probs)[1], len(look_ids)))
    for i in range(0, len(look_ids)):
        look_probs[:,i] = all_probs[np.where(catalog_ids == look_ids[i])[0]]
    #print test_ids
    class_names = ['M-Dwarf', 'Other Dwarf', 'Giant', 'Hot Star']
    look_class_names = np.asarray([class_names[np.argmax(look_probs[:,i])] for i in range(0,len(look_ids))])
    look_class_ids = np.asarray([np.argmax(look_probs[:,i]) for i in range(0,len(look_ids))])
    test_mask = np.where(look_class_names ==  class_names[0])
    print look_ids[test_mask]
    #print look_class_ids[test_mask]
    #print look_class_names[test_mask]
    #print look_probs[:,test_mask]
    new_mask = np.array([np.where(catalog_ids == look_ids[i])[0][0] for i in range(0,len(look_ids))])
    for k in range(0,len(test_mask[0])):
        print 'Kepid: ', int(look_ids[test_mask][k])
        print 'Impute Flags: ', impute_flags[new_mask[test_mask][k]]
        for j in range(0,len(goodcolids)):
            print colnames[goodcolnames[j]], impute_catalog[goodcolids[j],new_mask[test_mask][k]]
        print '\n'
    #print len(look_class_names)
    #print look_probs
