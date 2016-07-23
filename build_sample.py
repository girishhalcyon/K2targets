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
    condition = 100, goodcols = [1,8,9,10,13,14, 15],
    infoname = 'LoggJKcluster.npy', infocol = 1):

    infos = np.load(infoname)
    cluster_ids = infos[infocol][0]
    kepids = np.unique(cluster_ids)
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
    #if len(use_mask) > 100000:
    #    use_mask = use_mask[:100000]

    single_info = np.empty((len(goodcols)+1, len(infos[infocol][0])))
    single_details = np.empty((len(colnames)+1, len(infos[infocol][0])))
    i = 0
    for j in range(0,np.shape(infos)[1]):
        if j in goodcols:
            single_info[i,:] = np.asarray(infos[infocol][j])
            i +=1
        single_details[j,:] = np.asarray(infos[infocol][j])

    single_info[-1,:] = classlabel*np.ones_like(cluster_ids)
    single_details[-1,:] = classlabel*np.ones_like(cluster_ids)

    sample_info = np.empty((len(goodcols)+1, len(use_mask)))
    sample_details = np.empty((len(colnames)+1, len(use_mask)))

    for j in range(0,len(use_mask)):
        sample_info[:,j] = single_info[:,use_mask[j]]
        sample_details[:,j] = single_details[:,use_mask[j]]

    sample_info = sample_info.T
    sample_details = sample_details.T
    sample_info, sample_details = shuffle(sample_info, sample_details)
    return sample_info, sample_details

if __name__ == '__main__':
    colnames = ['KepID', 'KepMag', 'Teff', 'log g', 'Metallicity',
        'Mass', 'Radius', 'Distance', 'Jmag', 'Hmag', 'Kmag', 'Proper Motion',
        'Poly CDPP', 'gmag', 'rmag', 'imag', 'zmag', 'Class']
    goodcols = np.asarray([1,8,9,10,13,14, 15], dtype = int)
    #KepMag, J, H, K, PM, g, r, i, z
    teffname = 'TempCluster.npy'
    cdppname = 'CDPPcluster.npy'
    loggjkname = 'LoggJKcluster.npy'

    #M dwarfs: name=LoggJKcluster.npy; infocol=0; classname=0; occurence=100
    #Otherdwarfs: name=LoggJKcluster.npy; infocol=1; classname=1; occurence=100
    #Giants: name=CDPPcluster.npy; infocol=3; classname=2; occurence=100
    #Supergiants: name=CDPPcluster.npy; infocol=4; classname=3; occurence=100
    #Hot Stars: name=TempCluster.npy; infocol= 2; classname=4; occurence=100
    #Very Hot Stars: name=TempCluster.npy; infocol= 0; classname=5; occurence=100

    m_dwarf_sample, m_dwarf_details =  get_sample(classlabel = 0, infocol = 0,
        condition=100)
    subdwarf_sample, subdwarf_details = get_sample(classlabel= 1,infocol=1, condition=100)
    giant_sample, giant_details = get_sample(classlabel=2, infocol=0, infoname=cdppname, condition=100)
    #supergiant_sample, supergiant_details = get_sample(classlabel=2, infocol=4, infoname=cdppname, condition=100)
    hot_stars, hot_stars_details = get_sample(classlabel=3, infocol=2, infoname=teffname, condition=100)
    #very_hot_stars, very_hot_stars_details = get_sample(classlabel=3, infocol=0, infoname=teffname, condition=100)

    m_dwarf_train = int(round(float(np.shape(m_dwarf_sample)[0])*0.6))
    m_dwarf_calibrate = int(round(float(np.shape(m_dwarf_sample)[0])*0.8))
    subdwarf_train = int(round(float(np.shape(subdwarf_sample)[0])*0.6))
    subdwarf_calibrate = int(round(float(np.shape(subdwarf_sample)[0])*0.8))
    giant_train = int(round(float(np.shape(giant_sample)[0])*0.6))
    giant_calibrate = int(round(float(np.shape(giant_sample)[0])*0.8))
    #supergiant_train = int(round(float(np.shape(supergiant_sample)[0])*0.6))
    #supergiant_calibrate = int(round(float(np.shape(supergiant_sample)[0])*0.8))
    hot_train = int(round(float(np.shape(hot_stars)[0])*0.6))
    hot_calibrate = int(round(float(np.shape(hot_stars)[0])*0.8))
    #very_hot_train = int(round(float(np.shape(very_hot_stars)[0])*0.6))
    #very_hot_calibrate = int(round(float(np.shape(very_hot_stars)[0])*0.8))
    '''
    train_stack = np.vstack((m_dwarf_sample[:m_dwarf_calibrate],
        subdwarf_sample[:subdwarf_calibrate],
        giant_sample[:giant_calibrate], supergiant_sample[:supergiant_calibrate],
        hot_stars[:hot_calibrate], very_hot_stars[:very_hot_calibrate]))

    train_details = np.vstack((m_dwarf_details[:m_dwarf_calibrate],
        subdwarf_details[:subdwarf_calibrate],
        giant_details[:giant_calibrate], supergiant_details[:supergiant_calibrate],
        hot_stars_details[:hot_calibrate], very_hot_stars_details[:very_hot_calibrate]))

    calibrate_stack = np.vstack((m_dwarf_sample[m_dwarf_calibrate:],
        subdwarf_sample[subdwarf_calibrate:],
        giant_sample[giant_calibrate:],
        supergiant_sample[supergiant_calibrate:],
        hot_stars[hot_calibrate:],
        very_hot_stars[very_hot_calibrate:]))

    calibrate_details = np.vstack((m_dwarf_details[m_dwarf_calibrate:],
        subdwarf_details[subdwarf_calibrate:],
        giant_details[giant_calibrate:],
        supergiant_details[supergiant_calibrate:],
        hot_stars_details[hot_calibrate:],
        very_hot_stars_details[very_hot_calibrate:]))
    '''
    train_stack = np.vstack((m_dwarf_sample[:m_dwarf_calibrate],
        subdwarf_sample[:subdwarf_calibrate],
        giant_sample[:giant_calibrate],
        hot_stars[:hot_calibrate]))

    train_details = np.vstack((m_dwarf_details[:m_dwarf_calibrate],
        subdwarf_details[:subdwarf_calibrate],
        giant_details[:giant_calibrate],
        hot_stars_details[:hot_calibrate]))

    calibrate_stack = np.vstack((m_dwarf_sample[m_dwarf_calibrate:],
        subdwarf_sample[subdwarf_calibrate:],
        giant_sample[giant_calibrate:],
        hot_stars[hot_calibrate:]))

    calibrate_details = np.vstack((m_dwarf_details[m_dwarf_calibrate:],
        subdwarf_details[subdwarf_calibrate:],
        giant_details[giant_calibrate:],
        hot_stars_details[hot_calibrate:]))



    X_train = train_stack[:,:-1]
    y_train = train_stack[:,-1]
    #X_test = test_stack[:,:-1]
    #y_test = test_stack[:,-1]
    X_valid = calibrate_stack[:,:-1]
    y_valid = calibrate_stack[:,-1]
    np.save('X_train', X_train)
    #np.save('X_test', X_test)
    np.save('X_valid', X_valid)
    np.save('y_train',y_train)
    #np.save('y_test', y_test)
    np.save('y_valid',y_valid)
    #np.save('test_details', test_details)

    #X_train = np.load('X_train.npy')
    #y_train = np.load('y_train.npy')
    #X_test = np.load('X_test.npy')
    #y_test = np.load('y_test.npy')
    #X_valid = np.load('X_valid.npy')
    #y_valid = np.load('y_valid.npy')
    #test_details = np.load('test_details.npy')

    feature_names = [colnames[i] for i in goodcols]

    #print 'Imputing NaNs'
    #imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    #X_train = imp.fit_transform(X_train)
    #X_valid = imp.transform(X_valid)
    #X_test = imp.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, class_weight = 'balanced_subsample',oob_score=False,n_jobs = -1)

    print 'Training Forest'
    clf.fit(X_train, y_train)
    #print 'Predicting Test'
    #clf_probs = clf.predict_proba(X_test)
    print 'Calibrating'
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
    sig_clf.fit(X_valid, y_valid)
    #sig_clf_probs = sig_clf.predict_proba(X_test)
    #sig_score = log_loss(y_test, sig_clf_probs)

    feature_score =  clf.feature_importances_
    feature_score = feature_score/np.sum(feature_score)
    print 'Feature Importances (Normalized to Total):'
    for i in range(0,len(feature_names)):
        print feature_names[i], feature_score[i]
    #print 'Log-loss score:', sig_score
    #print 'Normal score:', sig_clf.score(X_test, y_test)
    #print 'Uncalibrated Normal score:', clf.score(X_test, y_test)
    joblib.dump(sig_clf, 'CalibratedRF_3.pkl')
    joblib.dump(clf, 'UncalibratedRF_3.pkl')
    '''


    plot_idx = 1
    n_classes = 6
    n_estimators = 80
    plot_colors = "ryb"
    cmap = plt.cm.RdYlBu
    plot_step = 0.02  # fine step width for decision surface contours
    plot_step_coarser = 0.5  # step widths for coarse classifier guesses
    numgoodcols = len(goodcols)
    uncal_score_chart = np.zeros((numgoodcols,numgoodcols))
    cal_score_chart = np.zeros((numgoodcols,numgoodcols))
    for i in range(0,7):
        for j in range(i+1,8):
            pair = [i,j]
        # We only take the two corresponding features
            X = X_train[:, pair]
            y = y_train
            X2 = np.vstack((X, X_valid[:,pair]))
            y2 = np.append(y, y_valid)
            X_calibrate = X_valid[:,pair]
            y_calibrate = y_valid
            X_test2 = X_test[:,pair]
            y_test2 = y_test
            # Shuffle
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]

            # Standardize
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            X = (X - mean) / std

            mean2 = X_test2.mean(axis=0)
            std2 = X_test2.std(axis=0)
            X_test2 = (X_test2-mean2)/std2

            mean3 = X_calibrate.mean(axis=0)
            std3 = X_calibrate.std(axis=0)
            X_calibrate = (X_calibrate-mean3)/std3
            # Train
            clf2 = clf
            clf2.fit(X2, y2)

            clf.fit(X, y)
            model = clf
            model2 = clf2
            #Calibrate
            sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
            sig_clf.fit(X_calibrate, y_calibrate)

            cal_scores = sig_clf.score(X_test2, y_test2)
            uncal_scores = clf2.score(X_test2, y_test2)
            cal_score_chart[i,j] = cal_scores
            cal_score_chart[j,i] = cal_scores
            uncal_score_chart[j,i] = uncal_scores
            uncal_score_chart[i,j] = uncal_scores
            # Create a title for each column and the console by using str() and
            # slicing away useless parts of the string
            model_title = 'Random Forest Classifier Calibrated'
            model_title2 = 'Random Forest Classifier Uncalibrated'
            model_details = model_title
            model_details2 = model_title2
            if hasattr(model, 'estimators_'):
                model_details += ' with {} estimators'.format(len(model.estimators_))
            print( model_details + ' with features: ' + feature_names[pair[0]] +', ' +feature_names[pair[1]] + ' has a score of '+ str(cal_scores) )
            if hasattr(model2, 'estimators_'):
                model_details2 += ' with {} estimators'.format(len(model2.estimators_))
            print( model_details2 + ' with features: ' + feature_names[pair[0]] +', ' +feature_names[pair[1]] + ' has a score of '+ str(uncal_scores) )

    labels = ['Kp', 'J', 'H', 'K', r'$\mu$', 'g', 'r', 'i', 'z']

    column_labels = labels
    row_labels = labels
    fig, ax = plt.subplots()
    data = uncal_score_chart
    heatmap = ax.pcolor(uncal_score_chart, cmap='Greys', vmin=0.0, vmax=1.0)
    fig.colorbar(heatmap)
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)

    ax.yaxis.grid(True, which='minor', linestyle='-', color='k', alpha = 0.5)
    ax.xaxis.grid(True, which='minor', linestyle='-', color='k', alpha = 0.5)

    # Set the location of the minor ticks to the edge of pixels for the x grid
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)

    # Lets turn off the actual minor tick marks though
    for tickmark in ax.xaxis.get_minor_ticks():
        tickmark.tick1On = tickmark.tick2On = False

    # Set the location of the minor ticks to the edge of pixels for the y grid
    minor_locator = AutoMinorLocator(2)
    ax.yaxis.set_minor_locator(minor_locator)

    # Lets turn off the actual minor tick marks though
    for tickmark in ax.yaxis.get_minor_ticks():
        tickmark.tick1On = tickmark.tick2On = False
    plt.axis('tight')
    plt.suptitle('Accuracy of RandomForestClassifier with Kepler Sample (Trained on 80%, Tested on 20%)')
    plt.savefig('RandomForestScoreUncal.pdf', dpi = 3000)

    plt.close('all')
    data = cal_score_chart
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(cal_score_chart, cmap='Greys', vmin=0.0, vmax=1.0)
    fig.colorbar(heatmap)
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)

    ax.yaxis.grid(True, which='minor', linestyle='-', color='k', alpha = 0.5)
    ax.xaxis.grid(True, which='minor', linestyle='-', color='k', alpha = 0.5)

    # Set the location of the minor ticks to the edge of pixels for the x grid
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)

    # Lets turn off the actual minor tick marks though
    for tickmark in ax.xaxis.get_minor_ticks():
        tickmark.tick1On = tickmark.tick2On = False

    # Set the location of the minor ticks to the edge of pixels for the y grid
    minor_locator = AutoMinorLocator(2)
    ax.yaxis.set_minor_locator(minor_locator)

    # Lets turn off the actual minor tick marks though
    for tickmark in ax.yaxis.get_minor_ticks():
        tickmark.tick1On = tickmark.tick2On = False
    plt.axis('tight')
    plt.suptitle('Accuracy of RandomForestClassifier with Kepler Sample (Trained on 60%, Calibrated on 20%, Tested on 20%)')
    plt.savefig('RandomForestScoreCal.pdf', dpi = 3000)
    '''
