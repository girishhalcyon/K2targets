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
    mass = tb_m_rad(rad)
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
    if teff > 0.0:
        return teff
    else:
        return np.nan

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
    if rad < 0.0:
        rad = np.nan
    return rad

def tb_m_rad(rad):
    if np.isfinite(rad):
        a = 0.32
        b = 0.6063
        c = 0.0906
        c = c - rad
        discrim = (b**2.0) - (4.0*a*c)
        polyparams = [a, b, c]
        if discrim < 0.0:
            return np.nan
        elif discrim == 0.0:
            return np.roots(polyparams)[0]
        elif discrim > 0.0:
            testroots = np.sort(np.roots(polyparams))
            root1 = testroots[0]
            root2 = testroots[1]
            if root1 < 0.0:
                return root2
            elif root2 > 1.5:
                return root1
            elif abs(0.5 - root1) <= abs(root2 - 0.5):
                return root1
            else:
                return root2
    else:
        return np.nan

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

    dwarfinfo = np.load('C5dwarfinfo.npy')
    giantinfo = np.load('C5giantinfo.npy')
    dginfo = np.load('C5dginfo.npy')
    infos = [giantinfo, dwarfinfo, dginfo]
    infotitles = ['Giants', 'Dwarfs', 'Both?']
    infocolors = ['r', 'k', 'c']
    infomarkers = ['o', 'x', '+']
    plottitle = 'C05'
    saveloc = 'C5plots/'


    #plot_epic_derive(infos, infotitles, infocolors, infomarkers,plottitle, saveloc = saveloc, yparam = 5, ylabel = r'T$_{eff}$')
    #plot_epic_derive(infos, infotitles, infocolors, infomarkers,plottitle, saveloc = saveloc, yparam = 7, ylabel = r'R$_{\star}$')
    plot_epic_derive(infos, infotitles, infocolors, infomarkers,plottitle, saveloc = saveloc, yparam = 8, ylabel = r'M$_{\star}$')
    '''

    plot_epic_derive(infos, infotitles, infocolors, infomarkers,plottitle, saveloc = saveloc, yparam = 5, ylabel = r'T$_{eff}$', mode = 'PNG')
    plot_epic_derive(infos, infotitles, infocolors, infomarkers,plottitle, saveloc = saveloc, yparam = 7, ylabel = r'R$_{\star}$', mode = 'PNG')
    plot_epic_derive(infos, infotitles, infocolors, infomarkers,plottitle, saveloc = saveloc, yparam = 8, ylabel = r'M$_{\star}$', mode = 'PNG')
    '''
