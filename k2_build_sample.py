import numpy as np
from ks17cluster import make_fuzzy_column
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from k2_verify import read_known, convert_sample_colors
from multiprocessing.dummy import Pool as ThreadPool

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
def make_test_sample(infos,goodcols = [1,7, 9,11, 13,15, 17, 19]):
    #KepMag, J, H, K, g, r, i, z
    #OR
    #KepMag, J, J-H, J-K, g, g-r, r-i, z-J
    kepids = infos[0]
    print np.shape(infos)
    single_info = np.empty((len(goodcols), len(infos[0])))
    print np.shape(single_info)
    #single_details = np.empty((np.shape(infos)[0]+1, len(infos[0])))
    #print np.shape(single_details)
    i = 0
    for j in range(0,np.shape(infos)[0]):
        if j in goodcols:
            single_info[i,:] = np.asarray(infos[j])
            i +=1
        #single_details[j,:] = np.asarray(infos[j])

    #single_details[-1,:] = 10.0*np.ones_like(kepids)

    sample_info = single_info.T
    #sample_details = single_details.T

    return sample_info


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

def main():


    colnames = ['KepID', 'KepMag', 'Teff', 'Metallicity',
        'Mass', 'Radius', 'Distance', 'Jmag', 'Hmag', 'Kmag', 'Proper Motion',
        'gmag', 'rmag', 'imag', 'zmag', 'Class']
    goodcolnames = np.asarray([1,7, 8,9,11,12, 13,14], dtype = int)
    goodcolids = [1,7, 9,11, 13,15, 17, 19]
    #goodcolnames = np.asarray([1,7, 8,9,11,12, 13], dtype = int)
    #goodcolids = [1,7, 9,11, 13,15, 17]
    feature_names = [colnames[i] for i in goodcolnames]

    C0_8_impute_catalog = np.load('C0_8_impute_catalog_color.npy')
    '''
    print 'Making Fuzzy Catalog'
    C0_8_fuzzy_catalog = make_fuzzy_catalog(C0_8_impute_catalog)
    np.save('C0_8_fuzzy_catalog_color', C0_8_fuzzy_catalog)

    C0_8_fuzzy_catalog = np.load('C0_8_fuzzy_catalog_color.npy')

    C0_8_K2_sample = make_test_sample(C0_8_fuzzy_catalog)
    np.save('C0_8_K2_sample_color',C0_8_K2_sample)
    #np.save('C0_8_K2_details_test', C0_8_K2_details)

    '''
    fuzz_factor = 100
    C0_8_impute_flags = np.load('C0_8_impute_flags_color.npy')
    C0_8_K2_sample = np.load('C0_8_K2_sample_color.npy')
    #C0_8_K2_sample = make_test_sample(C0_8_impute_catalog)
    #C0_8_K2_sample = convert_sample_colors(C0_8_K2_sample)
    #C0_8_K2_details = np.load('C0_8_K2_details.npy')
    print 'Loading Forest'
    clf = joblib.load('CalibratedRF_9.pkl')
    #n_classes = len(clf.classes_)
    n_classes = 4
    #classes = ['M-Dwarf', 'Other Dwarf', 'Giant', 'Hot Star']
    classes = ['Cool Dwarf', 'Giant', 'Hot','Other Dwarf']
    unique_len = np.shape(C0_8_K2_sample)[0]/fuzz_factor



    print 'Predicting K2 Data'
    probs = np.empty((unique_len, n_classes))

    prob_errs = np.empty_like(probs)

    all_probs = np.empty((unique_len*100, n_classes))
    def multi_prob_fill(i):
        all_probs[i*10000:(i+1)*10000] = clf.predict_proba(C0_8_K2_sample[i*10000:(i+1)*10000])
        print i*10000.0/16700000.0
    p = ThreadPool(None)
    p.map(multi_prob_fill, np.arange(0,unique_len/100))
    def multi_prob_calc(i):
        probs[i] = np.mean(all_probs[fuzz_factor*i:fuzz_factor*(i+1)], axis = 0)
        prob_errs[i] =  np.std(all_probs[fuzz_factor*i:fuzz_factor*(i+1)], axis = 0)
        print i+1
    p = ThreadPool(None)
    p.map(multi_prob_calc, np.arange(0,unique_len))
    np.save('C0_8_all_probs_CRF9', probs)
    np.save('C0_8_all_prob_errs_CRF9', prob_errs)

    #probs = clf.predict_proba(C0_8_K2_sample)
    #probs = clf.predict(C0_8_K2_sample[:,:-1])
    #np.save('C0_8_impute_probs3', probs)

if __name__ == '__main__':
    main()
