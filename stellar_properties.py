import numpy as np
import matplotlib.pyplot as plt


def test_entries(singletarg):
    Jmag = singletarg[19]
    if all([np.isfinite(Jmag), (Jmag >0.0)]):
        Hmag = singletarg[20]
        if all([np.isfinite(Hmag), (Hmag > 0.0)]):
            Vmag = singletarg[13]
            if all([np.isfinite(Vmag), (Vmag > 0.0)]):
                teff = am_vj_jh(Vmag, Jmag, Hmag)
                rad = am_r_teff(teff)
            else:
                rmag = singletarg[16]
                if all([np.isfinite(rmag), (rmag > 0.0)]):
                    teff = am_rj_jh(rmag, Jmag, Hmag)
                    rad = am_r_teff(teff)
                else:
                    teff = np.nan
                    rad = np.nan
        else:
            Vmag = singletarg[13]
            if all([np.isfinite(Vmag), (Vmag > 0.0)]):
                teff = am_vj(Vmag, Jmag)
                rad = am_r_teff(teff)
            else:
                rmag = singletarg[16]
                if all([np.isfinite(rmag), (rmag > 0.0)]):
                    teff = am_rj(rmag, Jmag)
                    rad = am_r_teff(teff)
                else:
                    teff = np.nan
                    rad = np.nan
    else:
        rmag = singletarg[16]
        if all([np.isfinite(rmag), (rmag > 0.0)]):
            zmag = singletarg[18]
            if all([np.isfinite(zmag), (zmag > 0.0)]):
                teff = am_rz(rmag, zmag)
                rad = am_r_teff(teff)
            else:
                teff = np.nan
                rad = np.nan
        else:
            teff = np.nan
            rad = np.nan
    mass = am_mk_mass(am_r_mk(rad))
    return teff, rad, mass

def am_vj_jh(Vmag, Jmag, Hmag):
    x = Vmag - Jmag
    y = Jmag - Hmag
    coeffs = [2.769, -1.421,  0.4284, -0.06133,  0.003310, 0.1333,  0.05416]
    teff = 3500.0*(coeffs[0] + coeffs[1]*x+ coeffs[2]*(x**2.0) + coeffs[3]*(x**3.0) + coeffs[4]*(x**4.0) + coeffs[5]*y + coeffs[6]*(y**2.0))
    if teff > 0.0:
        return teff
    else:
        return np.nan

def am_rj_jh(rmag, Jmag, Hmag):
    x = rmag - Jmag
    y = Jmag - Hmag
    coeffs = [2.151, -1.092, 0.3767, -0.06292,  0.003950,  0.1697,  0.03106]
    teff = 3500.0*(coeffs[0] + coeffs[1]*x+ coeffs[2]*(x**2.0) + coeffs[3]*(x**3.0) + coeffs[4]*(x**4.0) + coeffs[5]*y + coeffs[6]*(y**2.0))
    return teff

def am_vj(Vmag, Jmag):
    x = Vmag - Jmag
    coeffs = [2.840, -1.3453, 0.3906, -0.0546, 0.002913]
    teff = 3500.0*(coeffs[0] + coeffs[1]*x + coeffs[2]*(x**2.0) + coeffs[3]*(x**3.0) + coeffs[4]*(x**4.0))
    if teff > 0.0:
        return teff
    else:
        return np.nan

def am_rj(rmag, jmag):
    x= rmag - jmag
    coeffs = [2.445, -1.2578, 0.4340, -0.0720,  0.004502]
    teff = 3500.0*(coeffs[0] + coeffs[1]*x + coeffs[2]*(x**2.0) + coeffs[3]*(x**3.0) + coeffs[4]*(x**4.0))
    if teff > 0.0:
        return teff
    else:
        return np.nan

def am_rz(rmag, zmag):
    x = rmag - zmag
    coeffs = [1.547, -0.7053, 0.3656, -0.1008,  0.01046]
    teff = 3500.0*(coeffs[0] + coeffs[1]*x + coeffs[2]*(x**2.0) + coeffs[3]*(x**3.0) + coeffs[4]*(x**4.0))
    if teff > 0.0:
        return teff
    else:
        return np.nan

def am_r_teff(teff):
    x = teff/3500.0
    coeffs = [10.5440, -33.7546,  35.1909, -11.5928]
    rad = (coeffs[0] + coeffs[1]*x + coeffs[2]*(x**2.0) + coeffs[3]*(x**3.0))

    return rad

def am_r_mk(rad):
    coeffs = [1.9515, -0.3520, 0.01680]
    x = rad
    mk = (coeffs[0] + coeffs[1]*x + coeffs[2]*(x**2.0))

    return mk

def am_mk_mass(mk):
    coeffs = [0.5858, 0.3872, -0.1217, 0.0106, -0.00027262]
    x = mk
    mass = (coeffs[0] + coeffs[1]*x + coeffs[2]*(x**2.0) + coeffs[3]*(x**3.0) + coeffs[4]*(x**4.0))
    return mass

def tb_m_rad(rads):
    mass = np.empty_like(rads)
    for i in range(0,len(rads)):
        rad = rads[i]
        if np.isfinite(rad):
            a = 0.32
            b = 0.6063
            c = 0.0906
            c = c - rad
            discrim = (b**2.0) - (4.0*a*c)
            polyparams = [a, b, c]
            if discrim < 0.0:
                mass[i] = np.nan
            elif discrim == 0.0:
                mass[i] = np.roots(polyparams)[0]
            elif discrim > 0.0:
                testroots = np.sort(np.roots(polyparams))
                root1 = testroots[0]
                root2 = testroots[1]
                if root1 < 0.0:
                    if root2 > 0.0:
                        mass[i] = root2
                    else:
                        mass[i] = np.nan
                elif root2 > 1.5:
                    if root1 < 1.5:
                        mass[i] = root1
                    else:
                        mass[i] = np.nan
                elif abs(0.5 - root1) <= abs(root2 - 0.5):
                    mass[i] = root1
                else:
                    mass[i] = root2
        else:
            mass[i] = np.nan
    return mass
def plot_epic_derive(infos, infotitles, infocolors, infomarkers, plottitle,
    yparam, ylabel, xparam = 11, xlabel = 'KepMag',
    mode = 'SHOW', saveloc = ''):

    fig = plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    for k in range(0,len(infotitles)):
        info = infos[k]
        ax1.scatter(info[xparam], info[yparam], c = infocolors[k], marker = infomarkers[k], label=infotitles[k])
    ax1.legend(loc='best')



    for k in range(0,len(infotitles)):
        info = infos[k]
        teffs = np.empty((len(info[0])))
        rads = np.empty_like(teffs)
        masses = np.empty_like(rads)
        for i in range(0,len(info[0])):
            teffs[i], rads[i], masses[i] = test_entries(info[:,i])
        if yparam == 5:
            yvalues = teffs
        elif yparam == 7:
            yvalues = rads
        elif yparam == 8:
            yvalues = masses
        else:
            yvalues = info[yparam]

        if xparam == 5:
            xvalues = teffs
        elif xparam == 7:
            xvalues = rads
        elif xparam == 8:
            xvalues = masses
        else:
            xvalues = info[xparam]
        ax2.scatter(xvalues, yvalues, c = infocolors[k], marker = infomarkers[k], label = infotitles[k])

    plottitle2 = plottitle + ' ' + ylabel + ' vs ' + xlabel
    ax2.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax2.set_ylabel(ylabel)
    fig.suptitle(plottitle2)
    ax1.set_title('EPIC')
    ax2.set_title('Color Derived')
    if mode == 'SHOW':
        plt.show()
    else:
        if mode == 'PDF':
            filesuffix = '.pdf'
        else:
            filesuffix = '.png'

        plottitle2 = plottitle2.replace('$_{\star}$', 'star')
        plottitle2 = plottitle2.replace('$_{eff}$', 'eff')

        savetitle = plottitle2.replace(' ', '_')
        savein = saveloc + savetitle + filesuffix
        plt.savefig(savein)
    plt.close('all')

if __name__ == '__main__':
    colnames = ['KepID', 'KepMag', 'Teff', 'Metallicity',
        'Mass', 'Radius', 'Distance', 'Jmag', 'Hmag', 'Kmag', 'Proper Motion',
        'gmag', 'rmag', 'imag', 'zmag', 'Class']
    colnames = ['KepID', 'KepMag', 'Teff', 'Metallicity',
        'Mass', 'Radius', 'Distance', 'J', 'J - H', 'H - K', 'Proper Motion',
        'g', 'g - r', 'r - i', 'z - J', 'Class']

    impute_catalog_name = 'C0_8_impute_catalog_color'
    impute_catalog = np.load(impute_catalog_name + '.npy')
    catalog_ids = impute_catalog[0]
    catalog_J_mags = impute_catalog[7]
    catalog_H_mags = impute_catalog[7] - impute_catalog[9]
    catalog_r_mags = impute_catalog[13] - impute_catalog[15]
    catalog_teffs = am_rj_jh(catalog_r_mags, catalog_J_mags, catalog_H_mags)
    catalog_rads = am_r_teff(catalog_teffs)
    #catalog_mass = am_mk_mass(am_r_mk(catalog_rads))
    catalog_mass = tb_m_rad(catalog_rads)
    stell_props = np.empty((len(catalog_ids), 4))
    stell_props[:,0] = catalog_ids
    stell_props[:,1] = catalog_teffs
    stell_props[:,2] = catalog_rads
    stell_props[:,3] = catalog_mass
    stell_prop_name = impute_catalog_name + '_stell_props2'
    np.save(stell_prop_name, stell_props)
