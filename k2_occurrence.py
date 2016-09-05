import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stellar_properties import tb_m_rad as mass_calc
from scipy.stats import binom
from matplotlib.table import Table


def checkerboard_table(data, columns, rows, title, fmt='{:.2f}', logarea = 0.035):
    fig, ax = plt.subplots()
    plt.set_cmap('cool')
    ax.set_axis_off()

    tb = Table(ax, bbox=[0,0,1,1])
    #plt.suptitle(title)
    nrows, ncols = data.shape
    width, height = 1.0 / ncols, 1.0 / nrows
    min_color = np.min(data)/logarea
    max_color = np.max(data)/logarea
    # Add cells
    for (i,j), val in np.ndenumerate(data):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        color = (val/logarea)/max_color
        print val
        if val > 0.0:
            tb.add_cell(i, j, width, height, text=fmt.format(val),
                        loc='center', facecolor = str(1.0 - val/np.sum(data)))
        else:
            tb.add_cell(i, j, width, height)
    # Row Labels...
    for i in range(0,len(rows)):
        tb.add_cell(i, -1, width, height, text=rows[i], loc='top',
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j in range(0,len(columns)):
        tb.add_cell(-1, j, width, height/2.0, text=columns[j], loc='left',
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    #.colorbar()
    return fig

def snr_calc(R_p, P, R_s, noise, obs_dur):
    Jup_to_solar = 0.0995
    R_p = Jup_to_solar*R_p
    snr = (np.sqrt((obs_dur/P)))*(((R_p/R_s)**2.0)/noise)
    #print obs_dur/P
    return snr

def occurrence_error(n_planets, rate):
    try:
        n = n_planets/rate
        p = rate
        high = rate*n_planets/binom.ppf(0.159, n, p)
        low = rate*n_planets/binom.ppf(0.841, n, p)
        return rate - low,high - rate
    except:
        return np.nan, np.nan
def prob_tran_calc(R_star, P, M_star = [np.nan]):
    P = P/365.25
    if np.isfinite(M_star[0]):
        pass
    else:
        M_star = mass_calc(R_star)
    au_to_solar = 215.1
    semi_axis_cubed = ((P**2.0)*M_star)
    semi_axis = (semi_axis_cubed**(1.0/3.0))*au_to_solar
    return R_star/semi_axis

def N_obs_calc(R_p, P, all_R_s, all_noise, all_obs_dur, snr_floor = 10.0):
    all_snr = snr_calc(R_p, P, all_R_s, all_noise, all_obs_dur)
    #print np.max(all_snr)
    detected = all_snr[np.where(all_snr >= snr_floor)]
    return len(detected)

def prob_N_obs_calc(R_p, P, planet_prob, all_R_s, all_noise, all_obs_dur, snr_floor = 7.1*3.0, compensation_factor = 0.6):
    all_snr = snr_calc(R_p, P, all_R_s, all_noise, all_obs_dur)
    detected = all_snr[np.where(all_snr >= snr_floor)]
    inv_probs = 1.0/planet_prob
    #print planet_prob
    return (inv_probs)/(float(len(detected))*(compensation_factor))

def cdpp_calc(kepmags):
    ref_kepmags = np.load('kepmag_temp.npy')
    ref_cdpp = np.load('cdpp_temp.npy')
    cdpps = np.absolute(np.polyval(np.polyfit(ref_kepmags, ref_cdpp, 5), kepmags))
    return cdpps

def k2_noise_est(kepmags, cdpps):
    noise = np.copy(cdpps)
    for i in range(0,len(kepmags)):
        kepmag = kepmags[i]
        if kepmag < 11.0:
            noise[i] = 1.5*cdpps[i]
        elif kepmag < 14.0:
            noise[i] = 1.35*cdpps[i]
        else:
            noise[i] = 1.6*cdpps[i]
    return noise/(10.0**6.0)

def rate_bin(rads, periods, planet_probs, mask,all_R_s, all_noise, all_obs_dur, snr_floor=7.1, compensation_factor=2.0):
    occur_rate = 0.0
    for i in range(0,len(mask[0])):
        occur_rate += prob_N_obs_calc(rads[mask][i], periods[mask][i], planet_probs[mask][i], all_R_s, all_noise, all_obs_dur, snr_floor = snr_floor)
    return occur_rate*compensation_factor

def make_mask(rads, periods, rad_min, rad_max, period_min, period_max):
    return np.where((rad_min <= rads) & (rads < rad_max) & (period_min <= periods) & (periods < period_max))

if __name__ == '__main__':
    terra_candidates = pd.read_csv('terra_planets.csv')
    all_candidates = pd.read_csv('all_planets.csv')
    all_periods = np.array(all_candidates.P)
    all_Rp_E = np.array(all_candidates.Rp_E)
    all_Rp_J = np.array(all_candidates.Rp_J)
    all_planet_teffs = np.array(all_candidates.T_e)
    all_probs = np.array(all_candidates.Probs)
    terra_periods = np.array(terra_candidates.P)
    terra_teffs = np.array(terra_candidates.T_e)
    terra_probs = np.array(terra_candidates.Probs)
    terra_Rp_E = np.array(terra_candidates.Rp_E)
    terra_Rp_J = np.array(terra_candidates.Rp_J)
    terra_teff_hot = np.where(terra_teffs >= 3800.0)
    all_teff_hot = np.where(all_planet_teffs >= 3800.0)
    all_teff_cool = np.where(all_planet_teffs < 3800.0)
    terra_teff_cool = np.where(terra_teffs < 3800.0)

    fname = 'obs_star.csv'
    stell_df = pd.read_csv(fname)
    all_R_s = np.array(stell_df[' R_s'])
    #all_Kepmags = np.array(stell_df.Kepmag)
    all_noise = np.array(stell_df[' Noise'])/(10.0**6.0)
    all_obs_dur = np.array(stell_df[' Duration'])
    all_teffs = np.array(stell_df['T_e'])
    hot_mask = np.where(all_teffs >= 3800.0)
    cool_mask = np.where(all_teffs < 3800.0)



    rad_low = 0.001
    rad_high = 40.0
    period_low = 0.001
    period_high = 80.0

    period_mins = np.array([0.0, 10.0, 50.0])
    period_maxs = np.array([10.0, 50.0, 80.0])

    rad_mins = np.array([0.0, 1.4, 4.0,  8.0])
    rad_maxs = np.array([1.4, 4.0,  8.0, 50.0])
    n_period = len(period_mins)
    n_rad = len(rad_mins)
    terra_hot_table = np.empty((n_rad, n_period))
    terra_cool_table = np.empty_like(terra_hot_table)
    all_hot_table = np.empty_like(terra_hot_table)
    all_cool_table = np.empty_like(terra_hot_table)


    for i in range(0,n_period):
        for j in range(0,n_rad):

            mask_hot = make_mask(terra_Rp_E[terra_teff_hot], terra_periods[terra_teff_hot], rad_mins[j], rad_maxs[j],period_mins[i], period_maxs[i])
            mask_cool = make_mask(terra_Rp_E[terra_teff_cool], terra_periods[terra_teff_cool], rad_mins[j], rad_maxs[j],period_mins[i], period_maxs[i])
            temp_rate_hot = rate_bin(terra_Rp_J[terra_teff_hot], terra_periods[terra_teff_hot], terra_probs[terra_teff_hot], mask_hot, all_R_s[hot_mask], all_noise[hot_mask], all_obs_dur[hot_mask])
            temp_rate_cool = rate_bin(terra_Rp_J[terra_teff_cool], terra_periods[terra_teff_cool], terra_probs[terra_teff_cool], mask_cool, all_R_s[cool_mask], all_noise[cool_mask], all_obs_dur[cool_mask])
            temp_rate = temp_rate_hot + temp_rate_cool
            terra_hot_table[j,i] = temp_rate_hot
            terra_cool_table[j,i] = temp_rate_cool
            if temp_rate != 0.0000:
                print 'TERRA Hot, TERRA Cool:'
                temp_rate_hot_low, temp_rate_hot_high = occurrence_error(n_planets = len(mask_hot[0]), rate = temp_rate_hot)
                temp_rate_cool_low, temp_rate_cool_high = occurrence_error(n_planets = len(mask_cool[0]), rate = temp_rate_cool)
                temp_rate_low, temp_rate_high = occurrence_error(n_planets = len(mask_cool[0]) + len(mask_hot[0]), rate = temp_rate)
                print rad_mins[j], ' <= R_p < ', rad_maxs[j], '; ', period_mins[i], ' <= P < ', period_maxs[i], ': '

                print 'Hot:',temp_rate_hot, '+', temp_rate_hot_high, ' -', temp_rate_hot_low
                print 'N_planets hot: ', len(mask_hot[0])
                print 'Cool:',temp_rate_cool, '+', temp_rate_cool_high, ' -', temp_rate_cool_low
                print 'N_planets cool: ', len(mask_cool[0])
                print 'Total:',temp_rate, '+', temp_rate_high, ' -', temp_rate_low
                print 'N_planets: ', len(mask_hot[0]) + len(mask_cool[0])
                print '\n'

    terra_table = terra_hot_table + terra_cool_table
    #terra_hot_table[np.where(terra_hot_table < 0.00001)] = np.nan*terra_hot_table[np.where(terra_hot_table < 0.00001)]
    #terra_cool_table[np.where(terra_cool_table < 0.00001)] = np.nan*terra_cool_table[np.where(terra_cool_table < 0.00001)]
    #terra_table[np.where(terra_table < 0.00001)] = np.nan*terra_table[np.where(terra_table < 0.00001)]


    for i in range(0,n_period):
        for j in range(0,n_rad):
            mask_hot = make_mask(all_Rp_E[all_teff_hot], all_periods[all_teff_hot], rad_mins[j], rad_maxs[j],period_mins[i], period_maxs[i])
            mask_cool = make_mask(all_Rp_E[all_teff_cool], all_periods[all_teff_cool], rad_mins[j], rad_maxs[j],period_mins[i], period_maxs[i])
            temp_rate_hot = rate_bin(all_Rp_J[all_teff_hot], all_periods[all_teff_hot], all_probs[all_teff_hot], mask_hot,all_R_s[hot_mask], all_noise[hot_mask], all_obs_dur[hot_mask])
            temp_rate_cool = rate_bin(all_Rp_J[all_teff_cool], all_periods[all_teff_cool], all_probs[all_teff_cool], mask_cool, all_R_s[cool_mask], all_noise[cool_mask], all_obs_dur[cool_mask])
            temp_rate = temp_rate_hot + temp_rate_cool
            all_hot_table[j,i] = temp_rate_hot
            all_cool_table[j,i] = temp_rate_cool
            if temp_rate != 0.0000:
                print 'All Planets:'
                temp_rate_hot_low, temp_rate_hot_high = occurrence_error(n_planets = len(mask_hot[0]), rate = temp_rate_hot)
                temp_rate_cool_low, temp_rate_cool_high = occurrence_error(n_planets = len(mask_cool[0]), rate = temp_rate_cool)
                temp_rate_low, temp_rate_high = occurrence_error(n_planets = len(mask_cool[0]) + len(mask_hot[0]), rate = temp_rate)
                print rad_mins[j], ' <= R_p < ', rad_maxs[j], '; ', period_mins[i], ' <= P < ', period_maxs[i], ': '
                print 'Hot:',temp_rate_hot, '+', temp_rate_hot_high, ' -', temp_rate_hot_low
                print 'N_planets hot: ', len(mask_hot[0])
                print 'Cool:',temp_rate_cool, '+', temp_rate_cool_high, ' -', temp_rate_cool_low
                print 'N_planets cool: ', len(mask_cool[0])
                print 'Total:',temp_rate, '+', temp_rate_high, ' -', temp_rate_low
                print 'N_planets: ', len(mask_hot[0]) + len(mask_cool[0])


                print '\n'
    all_table = all_hot_table + all_cool_table
    #all_hot_table[np.where(all_hot_table < 0.00001)] = np.nan*all_hot_table[np.where(all_hot_table < 0.00001)]
    #all_cool_table[np.where(all_cool_table < 0.00001)] = np.nan*all_cool_table[np.where(all_cool_table < 0.00001)]
    #all_table[np.where(all_table < 0.00001)] = np.nan*all_table[np.where(all_table < 0.00001)]
    '''
    checkerboard_table(terra_hot_table, np.append([period_mins[0]], period_maxs), np.append([rad_mins[0]], rad_maxs), title = 'TERRA Hot Stars')
    plt.savefig('terra_hot.png')
    #plt.show()
    plt.clf()
    checkerboard_table(terra_cool_table, np.append([period_mins[0]], period_maxs), np.append([rad_mins[0]], rad_maxs), title = 'TERRA Cool Stars')
    plt.savefig('terra_cold.png')
    #plt.show()
    plt.clf()
    checkerboard_table(terra_table, np.append([period_mins[0]], period_maxs), np.append([rad_mins[0]], rad_maxs), title = 'TERRA')
    plt.savefig('terra_all.png')
    #plt.show()
    plt.clf()
    checkerboard_table(all_hot_table, np.append([period_mins[0]], period_maxs), np.append([rad_mins[0]], rad_maxs), title = 'ALL Hot Stars')
    plt.savefig('all_hot.png')
    #plt.show()
    plt.clf()
    checkerboard_table(all_cool_table, np.append([period_mins[0]], period_maxs), np.append([rad_mins[0]], rad_maxs), title = 'ALL Cool Stars')
    plt.savefig('all_cool.png')
    #plt.show()
    plt.clf()
    '''
    checkerboard_table(all_table, np.append([period_mins[0]], period_maxs), ['< Earth', '< Neptune', '< Jupiter', '> Jupiter'], title = 'ALL')
    plt.savefig('all_all.png')
    plt.show()
    plt.clf()
    '''
    xnum = 150
    ynum = 150
    Earth_to_Jup = 0.0922
    R_plans = np.logspace(np.log10(0.10*Earth_to_Jup), np.log10(10.0), xnum)
    Periods = np.logspace(np.log10(0.1), np.log10(80.0), ynum)

    gridx, gridy = np.meshgrid(R_plans, Periods)
    all_N_obs = np.empty((xnum, ynum))
    for i in range(0,xnum):
        for j in range(0,ynum):
            all_N_obs[i,j] = N_obs_calc(R_plans[i], Periods[j], all_R_s, all_noise, all_obs_dur)

    plt.pcolormesh(Periods, R_plans/Earth_to_Jup, all_N_obs/np.max(all_N_obs), cmap = 'bone_r')
    plt.plot(np.linspace(np.min(Periods), np.max(Periods), 1500), np.ones((1500)), '-b',label = 'Earth')
    plt.plot(np.linspace(np.min(Periods), np.max(Periods), 1500), 3.883*np.ones((1500)), '-g',label = 'Neptune')
    plt.plot(np.linspace(np.min(Periods), np.max(Periods), 1500), 11.209*np.ones((1500)), '-y',label = 'Jupiter')
    plt.plot(all_periods, all_Rp_E, 'or', label = 'K2 Planets')
    #plt.plot(terra_periods, terra_Rp_E, 'or', label = 'TERRA')
    xmin = np.min(Periods)
    xmax = np.max(Periods)
    ymin = np.min(R_plans/Earth_to_Jup)
    ymax = np.max(R_plans/Earth_to_Jup)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel('Period [days]')
    plt.ylabel(r'R$_p$ [R$_\oplus$]')
    plt.title(r'Detectable N$_{transits}$')
    plt.colorbar(label='Fraction of Detectable Red Dwarfs')
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('detectable_transits.png')
    plt.show()
    plt.clf()

    plt.pcolormesh(Periods, R_plans/Earth_to_Jup, all_N_obs, cmap = 'bone_r')
    plt.plot(all_periods[all_teff_hot], all_Rp_E[all_teff_hot], 'sb', label = '> 3800 K')
    plt.plot(all_periods[all_teff_cool], all_Rp_E[all_teff_cool], 'or', label = '<= 3800 K')
    xmin = np.min(Periods)
    xmax = np.max(Periods)
    ymin = np.min(R_plans/Earth_to_Jup)
    ymax = np.max(R_plans/Earth_to_Jup)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel('Period [days]')
    plt.ylabel(r'R$_p$ [R$_\oplus$]')
    plt.title(r'Detectable N$_{transits}$')
    plt.colorbar()
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('hot_cool_detectable_transits.png')
    #plt.show()
    '''
