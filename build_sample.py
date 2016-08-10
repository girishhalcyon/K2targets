import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.preprocessing import Imputer
from sklearn.utils import shuffle
from treeinterpreter import treeinterpreter as ti
from matplotlib.ticker import AutoMinorLocator
from sklearn.externals import joblib

def get_sample(classlabel = 0,
    condition = 100, goodcols = [1,8,9,10,13,14, 15, 16],
    infoname = 'LoggJKcluster.npy', infocol = 1):

    infos = np.load(infoname)
    cluster_ids = infos[infocol][0]
    print np.mean(infos[infocol][3])
    kepids = np.unique(cluster_ids)
    print len(kepids)
    occurences = np.empty((len(kepids)), dtype = int)
    for i in range(0,len(kepids)):
        nkep = len(np.where(cluster_ids == kepids[i])[0])
        occurences[i] = nkep
    if condition == 100:
        sample_kepids = kepids[np.where(occurences == 100)]
    else:
        sample_kepids = kepids[np.where(occurences >= condition)]
    keep_mask = np.empty((len(cluster_ids)))

    for i in range(0,len(cluster_ids)):
        if cluster_ids[i] in sample_kepids:
            keep_mask[i] = True
        else:
            keep_mask[i] = False
    use_mask = np.where(keep_mask)[0]
    np.random.shuffle(use_mask)
    if classlabel != 0:
        if len(use_mask) > 16800:
            use_mask = use_mask[:16800]

    single_info = np.empty((len(goodcols)+1, len(infos[infocol][0])))

    i = 0
    for j in range(0,np.shape(infos)[1]):
        if j in goodcols:
            single_info[i,:] = np.asarray(infos[infocol][j])
            i +=1


    single_info[-1,:] = classlabel*np.ones_like(cluster_ids)


    sample_info = np.empty((len(goodcols)+1, len(use_mask)))


    for j in range(0,len(use_mask)):
        sample_info[:,j] = single_info[:,use_mask[j]]


    sample_info = sample_info.T

    sample_info = shuffle(sample_info)
    return sample_info


def sample_color_convert(sample_info):
    color_sample = np.copy(sample_info)
    Kepmags = sample_info[:,0]
    Jmags = sample_info[:,1]
    Hmags = sample_info[:,2]
    Kmags = sample_info[:,3]
    gmags = sample_info[:,4]
    rmags = sample_info[:,5]
    imags = sample_info[:,6]
    zmags = sample_info[:,7]

    #KepMag, J, H, K, g, r, i, z: input
    #KepMag, J, J-H, H-K, g, g - r, r - i, z - J: output
    color_sample[:,0] = Kepmags
    color_sample[:,1] = Jmags
    color_sample[:,2] = Jmags - Hmags
    color_sample[:,3] = Hmags - Kmags
    color_sample[:,4] = gmags
    color_sample[:,5] = gmags - rmags
    color_sample[:,6] = rmags - imags
    color_sample[:,7] = zmags - Jmags
    return color_sample

if __name__ == '__main__':
    colnames = ['KepID', 'KepMag', 'Teff', 'log g', 'Metallicity',
        'Mass', 'Radius', 'Distance', 'Jmag', 'Hmag', 'Kmag', 'Proper Motion',
        'Poly CDPP', 'gmag', 'rmag', 'imag', 'zmag', 'Class']
    goodcols = np.asarray([1,8,9,10,13,14, 15, 16], dtype = int)
    #KepMag, J, H, K, g, r, i, z
    teffname = 'TempCluster.npy'
    cdppname = 'CDPPcluster.npy'
    loggjkname = 'LoggJKcluster.npy'

    #M dwarfs: name=LoggJKcluster.npy; infocol=0; classname=0; occurence=100
    #Otherdwarfs: name=LoggJKcluster.npy; infocol=1; classname=1; occurence=100
    #Giants: name=CDPPcluster.npy; infocol=3; classname=2; occurence=100
    #Supergiants: name=CDPPcluster.npy; infocol=4; classname=3; occurence=100
    #Hot Stars: name=TempCluster.npy; infocol= 2; classname=4; occurence=100
    #Very Hot Stars: name=TempCluster.npy; infocol= 0; classname=5; occurence=100

    m_dwarf_sample =  get_sample(classlabel = 0, infocol = 0,
        condition=80)
    subdwarf_sample= get_sample(classlabel= 3,infocol=1, condition=100)
    giant_sample = get_sample(classlabel=1, infocol=0, infoname=cdppname, condition=100)
    hot_stars = get_sample(classlabel=2, infocol=2, infoname=teffname, condition=100)

    m_dwarf_sample = sample_color_convert(m_dwarf_sample)
    subdwarf_sample = sample_color_convert(subdwarf_sample)
    giant_sample = sample_color_convert(giant_sample)
    hot_stars = sample_color_convert(hot_stars)

    m_dwarf_train = int(round(float(np.shape(m_dwarf_sample)[0])*0.6))
    m_dwarf_calibrate = int(round(float(np.shape(m_dwarf_sample)[0])*0.8))
    subdwarf_train = int(round(float(np.shape(subdwarf_sample)[0])*0.6))
    subdwarf_calibrate = int(round(float(np.shape(subdwarf_sample)[0])*0.8))
    giant_train = int(round(float(np.shape(giant_sample)[0])*0.6))
    giant_calibrate = int(round(float(np.shape(giant_sample)[0])*0.8))

    hot_train = int(round(float(np.shape(hot_stars)[0])*0.6))
    hot_calibrate = int(round(float(np.shape(hot_stars)[0])*0.8))


    train_stack = np.vstack((m_dwarf_sample[:m_dwarf_calibrate],
        subdwarf_sample[:subdwarf_calibrate],
        giant_sample[:giant_calibrate],
        hot_stars[:hot_calibrate]))



    calibrate_stack = np.vstack((m_dwarf_sample[m_dwarf_calibrate:],
        subdwarf_sample[subdwarf_calibrate:],
        giant_sample[giant_calibrate:],
        hot_stars[hot_calibrate:]))





    X_train = train_stack[:,:-1]
    y_train = train_stack[:,-1]
    #X_test = test_stack[:,:-1]
    #y_test = test_stack[:,-1]
    X_valid = calibrate_stack[:,:-1]
    y_valid = calibrate_stack[:,-1]
    '''
    np.save('X_train', X_train)
    #np.save('X_test', X_test)
    np.save('X_valid', X_valid)
    np.save('y_train',y_train)
    #np.save('y_test', y_test)
    np.save('y_valid',y_valid)
    #np.save('test_details', test_details)

    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    #X_test = np.load('X_test.npy')
    #y_test = np.load('y_test.npy')
    X_valid = np.load('X_valid.npy')
    y_valid = np.load('y_valid.npy')
    '''
    #test_details = np.load('test_details.npy')

    feature_names = [colnames[i] for i in goodcols]
    #feature_names = ['KepMag', 'J', 'J - H', 'H - K', 'g', 'g - r', 'r - i', 'z - J']


    clf = RandomForestClassifier(n_estimators=1000, max_features = 3,class_weight = 'balanced_subsample',oob_score=True,n_jobs = -1)

    print 'Training Forest'
    clf.fit(X_train, y_train)

    print 'Calibrating'
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
    sig_clf.fit(X_valid, y_valid)


    feature_score =  clf.feature_importances_
    feature_score = feature_score/np.sum(feature_score)
    print 'Feature Importances (Normalized to Total):'
    for i in range(0,len(feature_names)-1):
        print feature_names[i], feature_score[i]

    joblib.dump(sig_clf, 'CalibratedRF_9.pkl')
    joblib.dump(clf, 'UncalibratedRF_9.pkl')
