import numpy as np
import pandas as pd
#import csv

if __name__ == '__main__':
    #Replace epicpath  with your own filepath for the epic csv#
    #The epic csv can be downloaded from MAST at "https://archive.stsci.edu/k2/catalogs.html"#

    epicpath = 'C:\Users\Girish\Desktop\k2_epic.csv'

    #Uncomment l16-17 + l3: import csv #
    #if you want to list all the column headers for the epic csv#
    #More information on the column header can be found at: "https://archive.stsci.edu/missions/k2/catalogs/README_epic"#
    #Note that the capitalization in the README differs slightly from the true file column headers#

    #test = csv.DictReader(open(epicpath))
    #print test.fieldnames


    #Using docstrings, comment out the majority of this program.#
    #When running, holding each column of the csv in a pandas DataFrame ~ 0.35 GB#
    #So, depending on total RAM, must limit number of columns read at a time#
    #2 sections per run is good for my laptop (1 section = 2 columns)#
    #Comment out all sections other than the ones you want to run#
    #Unfortunately, writing to disk takes time, ~10 minutes per 2 section run#
    '''
    k2csv = pd.read_csv(epicpath, usecols = ['k2_ra', 'k2_dec'])
    ras = np.asarray(k2csv['k2_ra'])
    decs = np.asarray(k2csv['k2_dec'])
    np.save('ras', ras)
    np.save('decs', decs)

    k2csv = pd.read_csv(epicpath, usecols = ['pmRA', 'pmDEC'])
    ras = np.asarray(k2csv['pmRA'])
    decs = np.asarray(k2csv['pmDEC'])
    np.save('pmRAs', pmRAs)
    np.save('pmDecs', pmDecs)

    k2csv = pd.read_csv(epicpath, usecols = ['bmag', 'vmag'])
    Bmags = np.asarray(k2csv['bmag'])
    Vmags = np.asarray(k2csv['vmag'])
    np.save('Bmags', Bmags)
    np.save('Vmags', Vmags)

    k2csv = pd.read_csv(epicpath, usecols = ['umag', 'gmag'])
    umags = np.asarray(k2csv['umag'])
    gmags = np.asarray(k2csv['gmag'])
    np.save('umags', umags)
    np.save('gmags', gmags)

    k2csv = pd.read_csv(epicpath, usecols = ['rmag', 'imag'])
    rmags = np.asarray(k2csv['rmag'])
    imags = np.asarray(k2csv['imag'])
    np.save('rmags', rmags)
    np.save('imags', imags)

    k2csv = pd.read_csv(epicpath, usecols = ['zmag', 'jmag'])
    zmags = np.asarray(k2csv['zmag'])
    Jmags = np.asarray(k2csv['jmag'])
    np.save('zmags', zmags)
    np.save('Jmags', Jmags)

    k2csv = pd.read_csv(epicpath, usecols = ['hmag', 'kmag'])
    Hmags = np.asarray(k2csv['hmag'])
    Kmags = np.asarray(k2csv['kmag'])
    np.save('Hmags', Hmags)
    np.save('Kmags', Kmags)

    k2csv = pd.read_csv(epicpath, usecols = ['w1mag', 'w2mag'])
    W1mags = np.asarray(k2csv['w1mag'])
    W2mags = np.asarray(k2csv['w2mag'])
    np.save('W1mags', W1mags)
    np.save('W2mags', W2mags)


    k2csv = pd.read_csv(epicpath, usecols = ['w3mag', 'w4mag'])
    W3mags = np.asarray(k2csv['w3mag'])
    W4mags = np.asarray(k2csv['w4mag'])
    np.save('W3mags', W3mags)
    np.save('W4mags', W4mags)

    k2csv = pd.read_csv(epicpath, usecols = ['kp', 'teff'])
    kepmags = np.asarray(k2csv['kp'])
    Teffs = np.asarray(k2csv['teff'])
    np.save('kepmags', kepmags)
    np.save('Teffs', Teffs)
    ''''
    k2csv = pd.read_csv(epicpath, usecols = ['logg', 'feh'])
    loggs = np.asarray(k2csv['logg'])
    metals = np.asarray(k2csv['feh'])
    np.save('loggs', loggs)
    np.save('metals', metals)

    k2csv = pd.read_csv(epicpath, usecols = ['rad', 'mass'])
    Rads = np.asarray(k2csv['rad'])
    masses = np.asarray(k2csv['mass'])
    np.save('Rads', Rads)
    np.save('masses', masses)
    '''
    k2csv = pd.read_csv(epicpath, usecols = ['rho', 'lum'])
    rhos = np.asarray(k2csv['rho'])
    Lums = np.asarray(k2csv['lum'])
    np.save('rhos', rhos)
    np.save('Lums', Lums)

    k2csv = pd.read_csv(epicpath, usecols = ['d', 'ebv'])
    Dists = np.asarray(k2csv['d'])
    EBVs = np.asarray(k2csv['ebv'])
    np.save('Dists', Dists)
    np.save('EBVs', EBVs)
    '''
