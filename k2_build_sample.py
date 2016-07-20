import numpy as np
from ks17cluster import make_fuzzy_column
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt

def make_test_sample(infos,goodcols = [1,7, 9,11, 13,15, 17, 19]):
    #KepMag, J, H, K, PM, g, r, i, z
    kepids = infos[0]
    print np.shape(infos)
    single_info = np.empty((len(goodcols), len(infos[0])))
    print np.shape(single_info)
    single_details = np.empty((np.shape(infos)[0]+1, len(infos[0])))
    print np.shape(single_details)
    i = 0
    for j in range(0,np.shape(infos)[0]):
        if j in goodcols:
            single_info[i,:] = np.asarray(infos[j])
            i +=1
        single_details[j,:] = np.asarray(infos[j])

    single_details[-1,:] = 10.0*np.ones_like(kepids)

    sample_info = single_info.T
    sample_details = single_details.T

    return sample_info, sample_details

def make_fuzzy_catalog(impute_catalog,fuzz_factor = 100):
    fuzzy_catalog = np.empty((np.shape(impute_catalog)[0], np.shape(impute_catalog)[1]*fuzz_factor))
    no_err = [0, 1, 2, 3, 4, 5, 6]
    yes_err = [7, 9, 11, 13, 15, 17, 19, 21]
    err_cols = [8, 10, 12, 14, 16, 18, 20, 22]
    for i in no_err:
        fuzzy_catalog[i] = make_fuzzy_column(impute_catalog[i])
    for j in yes_err:
        print j
        try:
            fuzzy_catalog[j] = make_fuzzy_column(impute_catalog[j], impute_catalog[j+1])
        except:
            fuzzy_catalog[j] = make_fuzzy_column(impute_catalog[j], [0.003]*len(impute_catalog[j]))
    for k in err_cols:
        fuzzy_catalog[k] = make_fuzzy_column(impute_catalog[k])
    return fuzzy_catalog

if __name__ == '__main__':

    colnames = ['KepID', 'KepMag', 'Teff', 'Metallicity',
        'Mass', 'Radius', 'Distance', 'Jmag', 'Hmag', 'Kmag', 'Proper Motion',
        'gmag', 'rmag', 'imag', 'zmag', 'Class']
    goodcolnames = np.asarray([1,7, 8,9,11,12, 13,14], dtype = int)
    goodcolids = [1,7, 9,11, 13,15, 17, 19]
    feature_names = [colnames[i] for i in goodcolnames]

    C6_impute_catalog = np.load('C6_impute_catalog.npy')

    print 'Making Fuzzy Catalog'
    C6_fuzzy_catalog = make_fuzzy_catalog(C6_impute_catalog)
    np.save('C6_fuzzy_catalog', C6_fuzzy_catalog)
    C6_fuzzy_catalog = np.load('C6_fuzzy_catalog.npy')

    C6_K2_sample, C6_K2_details = make_test_sample(C6_fuzzy_catalog)
    np.save('C6_K2_sample',C6_K2_sample)
    np.save('C6_K2_details', C6_K2_details)


    fuzz_factor = 100
    C6_impute_flags = np.load('C6_impute_flags.npy')
    C6_K2_sample = np.load('C6_K2_sample.npy')
    C6_K2_details = np.load('C6_K2_details.npy')
    print 'Loading Forest'
    clf = joblib.load('CalibratedRF_2.pkl')
    n_classes = len(clf.classes_)
    classes = ['M-Dwarf', 'Other Dwarf', 'Giant', 'Hot Star']
    unique_len = np.shape(C6_K2_sample)[0]/fuzz_factor



    print 'Predicting K2 Data'
    probs = np.empty((unique_len, n_classes))
    for i in range(0,unique_len):
        probs[i] = np.mean(clf.predict_proba(C6_K2_sample[fuzz_factor*i:fuzz_factor*(i+1)]), axis = 0)
        print i, ' out of ', unique_len
    np.save('C6_all_probs_CRF2', probs)
