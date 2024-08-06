"""
This file is part of the accompanying code to our paper: Jiang, S., Tarasova, L., Yu, G., & Zscheischler, J. (2024). Compounding effects in flood drivers challenge estimates of extreme river floods. Science Advances, 10(13), eadl4005.

Copyright (c) 2024 Shijie Jiang. All rights reserved.
You should have received a copy of the MIT license along with the code. If not,
see <https://opensource.org/licenses/MIT>
"""

import proplot as pplt
import numpy as np
import scipy as sp
from matplotlib.lines import Line2D


def plot_compounding_driver(pd_all_peak_all_analysis, pd_am_peak_all_analysis, all_threshold_list, n_repeats):

    fig, axs = pplt.subplots([[1, 2, 3, 4], [5, 5, 5, 5], [6, 6, 6, 6]], refheight=1.5, refwidth=0.8,
                            hspace=[6, 0.5], hratios=[0.8, 0.5, 0.3], wspace=1,
                            share=True, span=False, grid=False, abc=False)

    var_order = ['rr', 'tg', 'sm', 'sp']
    cmap_rd = pplt.Colormap('Reds', right=0.6)
    cmap_bl = pplt.Colormap('Blues', right=0.6)
    cmap_gn = pplt.Colormap('Greens', right=0.6)
    cmap_pp = pplt.Colormap('Purples', right=0.6)

    cmaps_ = [cmap_bl, cmap_rd, cmap_gn, cmap_pp]

    feature_names_display = ['Precipitation (mm)', 'Temperature (°C)', 'Soil moisture (mm)', 'Snowpack (mm)']
    feature_names_display = ['$\overline{RR}$ (mm)', '$\overline{RT}$ (°C)', '$SM$ (mm)', '$SP$ (mm)']

    scs = []

    for i, ax in enumerate(axs[0:4]):
        ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
        all_peak_to_plot = pd_all_peak_all_analysis.sort_values(
            f'{var_order[i]}_value')

        sc1 = ax.scatter(x=all_peak_to_plot[f'{var_order[i]}_value'],
                         y=all_peak_to_plot[f'{var_order[i]}_shap_q50'],
                         markersize=4.0, color='#bbbbbb', marker='o', alpha=1.0)
        ###############################################################################
        am_peak_to_plot = pd_am_peak_all_analysis.sort_values(
            f'{var_order[i]}_value')
        ax.errorbar(x=am_peak_to_plot[f'{var_order[i]}_value'],
                    y=am_peak_to_plot[f'{var_order[i]}_shap_q50'],
                    yerr=am_peak_to_plot[[
                        f'{var_order[i]}_shap_q90_diff', f'{var_order[i]}_shap_q10_diff']].values.T,
                    capsize=0, lw=0.6, c='k', alpha=0.5, fmt='none',
                    )

        ax.axhline(np.median(all_threshold_list),
                   color='red7', linewidth=1.0, linestyle='--')
        ax.axhspan(np.quantile(all_threshold_list, 0.05), np.quantile(
            all_threshold_list, 0.95), facecolor='red7', alpha=0.2, closed=False)

        sc2 = ax.scatter(x=am_peak_to_plot[f'{var_order[i]}_value'],
                         y=am_peak_to_plot[f'{var_order[i]}_shap_q50'],
                         c=am_peak_to_plot[f'{var_order[i]}_imp_sum'], vmin=0, vmax=n_repeats,
                         markersize=12, marker='o', zorder=30, ec='k', lw=0.5, cmap=cmaps_[i])
        scs.append(sc2)

        if i == 0:
            ax.format(yloc=('outward', 3), xloc='bottom', xlabel=feature_names_display[i], xlabel_kw={
                      'va': 'bottom'}, xlabelpad=12, tickdir='in', ticklen=2, tickwidth=0.5, lw=0.5, ticklabelsize=8.5)

        else:
            ax.format(yloc='none', xloc='bottom', xlabel=feature_names_display[i], xlabel_kw={
                      'va': 'bottom'}, xlabelpad=12, tickdir='in', ticklen=2, tickwidth=0.5, lw=0.5, ticklabelsize=8.5)
            ax.format(ygrid=False, xgrid=False, ylabel='Aggregated contribution', ylabel_kw={
                      'va': 'top'}, tickdir='in', ticklen=2, tickwidth=0.5, lw=0.5, ticklabelsize=8.5)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='non-AM discharges', markerfacecolor='#bbbbbb', markersize=5),
                       Line2D([0], [0], marker='o', color='w', label='AM discharges', markerfacecolor='white', markeredgecolor='k', markersize=4.5, lw=0.1)]

    axs[2].legend(legend_elements, ncol=1, frame=False, loc='i')

    ax = axs[4]
    pd_am_peak_all_analysis_sorted = pd_am_peak_all_analysis.sort_values('y_true')

    num_envents = len(pd_am_peak_all_analysis_sorted)

    ax.pcolormesh(x=np.arange(num_envents+1)-0.5, y=[0.05, 0.95], z=pd_am_peak_all_analysis_sorted[['rr_imp_sum']].values, 
                  transpose=True, ec='w', lw=0.5, vmax=n_repeats, vmin=0, cmap=cmap_bl)
    ax.pcolormesh(x=np.arange(num_envents+1)-0.5, y=[1.05, 1.95], z=pd_am_peak_all_analysis_sorted[['tg_imp_sum']].values, 
                  transpose=True, ec='w', lw=0.5, vmax=n_repeats, vmin=0, cmap=cmap_rd)
    ax.pcolormesh(x=np.arange(num_envents+1)-0.5, y=[2.05, 2.95], z=pd_am_peak_all_analysis_sorted[['sm_imp_sum']].values,
                  transpose=True, ec='w', lw=0.5, vmax=n_repeats, vmin=0, cmap=cmap_gn)
    ax.pcolormesh(x=np.arange(num_envents+1)-0.5, y=[3.05, 3.95], z=pd_am_peak_all_analysis_sorted[['sp_imp_sum']].values, 
                  transpose=True, ec='w', lw=0.5, vmax=n_repeats, vmin=0, cmap=cmap_pp)

    ax.format(grid=False, xtickloc='neither', xlabel='', ylim=(4, 0), yticks=[3.5, 2.5, 1.5, 0.5], yticklabels=['$SP$', '$SM$', '$RT$', '$RR$'], yloc=('outward', 3), xloc='none',
              tickdir='in', ticklen=2, tickwidth=0.5, lw=0.5, ticklabelsize=8.5)
    ###################################################################################################
    ax = axs[5]
    ax.plot(pd_am_peak_all_analysis_sorted['y_true'].values, lw=1, ls='--',
            color='gray6', marker='o', markersize=3.5, markercolor='k', markeredgewidth=0)
    ax.format(yloc=('outward', 3), xloc=('outward', 15), xlabel='Rank', xticklabelloc='neither', ylim=(-1, 250), yticks=[50, 100, 200],
              ylabel='Discharge\n(m$^3$s$^{-1}$)', ylabelpad=1, yscale='log', tickdir='in', ticklen=2, tickwidth=0.5, lw=0.5, ticklabelsize=8.5)

    ax.annotate("", xy=(1, -0.55), xytext=(0.9, -0.55),
                arrowprops={"arrowstyle": "-|>", "color": "k"}, xycoords='axes fraction')
    ax.annotate("more extreme", xy=(1.00, -0.8),
                xycoords='axes fraction', ha='right', style='oblique')

    len_x = len(pd_am_peak_all_analysis_sorted)

    def forward(x):
        return (1 / ((len_x - x) / (len_x + 1)))

    def inverse(x):
        return (len_x - 1 / x * (len_x + 1))

    se_ax = ax.inset((0.5, 0.02, 0.5, 0.05), transform='axes', zoom=False)
    se_ax.plot(pd_am_peak_all_analysis_sorted['y_true'].values, color='none')
    se_ax.format(grid=False, xloc='none', yloc='none', tickdir='in',
                 ticklen=2, tickwidth=0.5, lw=0.5, ticklabelsize=8.5)
    se_ax = se_ax.dualx((forward, inverse), loc=('axes', 0))
    se_ax.format(xticks=[2, 5, 10, 20], xminorticks='none', xlabel='Return period (year)', xlim=(2, None), xlabelloc='top', xticklabelsize='med', xlabelpad=2, xlabelsize='med',
                 tickdir='in', ticklen=2, tickwidth=0.5, lw=0.5, ticklabelsize=8.5)


def plot_flood_complexity(pd_am_peak_eval_interaction, com_linear_inter, com_linear_slope, com_linear_pvalu, n_repeats):
    
    com_linear_inter_mean = np.median(com_linear_inter)
    com_linear_slope_mean = np.median(com_linear_slope)
    
    fig, ax = pplt.subplots(refwidth=2.5, refheight=1.8)

    eb = ax.errorbar(pd_am_peak_eval_interaction['non exceedance probability'], pd_am_peak_eval_interaction['num_imp_q50'], 
            yerr=pd_am_peak_eval_interaction[['num_imp_q10_diff', 'num_imp_q90_diff']].values.T,
            fmt='none', capsize=0, lw=1, c='k', alpha=0.5)
    eb[-1][0].set_linestyle((0, (1, 1)))
    ax.scatter(pd_am_peak_eval_interaction['non exceedance probability'], pd_am_peak_eval_interaction['num_imp_q50'], color='k', s=8)


    for i in range(n_repeats):
        ax.plot([0, 100], [com_linear_inter[i], com_linear_inter[i]+com_linear_slope[i]*100], color='orange5', lw=0.5, ls='-', alpha=0.3)

    ax.plot([0, 100], [com_linear_inter_mean, com_linear_inter_mean+com_linear_slope_mean*100], color='tomato', lw=2.5)
        
    ax.text(0.03, 0.93, f'Flood complexity = {com_linear_slope_mean:0.3f}', color='tomato', transform='axes')
    ax.text(0.03, 0.84, f"(combined $p$ = {sp.stats.combine_pvalues(com_linear_pvalu, method='fisher')[1]:0.2g})", color='tomato', transform='axes')
    ax.format(xlim=[0, 100], xlabel='Non-exceedance probability (%)', ylabel='Interaction richness (%)', xloc=('outward', 15), yloc=('outward', 3),
            ylim=(4, 26), tickdir='in', ticklen=2, tickwidth=0.5, lw=0.5, ticklabelsize=8.5)

    def forward(x):
        return 1 / (100 - x) * 100

    def inverse(x):
        return 100 - 1 / (x / 100)

    se_ax = ax.inset((0.5, 0.00, 0.5, 0.05), transform='axes', zoom=False)
    se_ax.plot(pd_am_peak_eval_interaction['non exceedance probability'], pd_am_peak_eval_interaction['num_imp_q50'], color='none')
    se_ax.format(grid=False, xloc='none', yloc='none', xlabel='', ylabel='', xlim=[50, 99.99999])
    se_ax = se_ax.dualx((forward, inverse), loc=('axes', 0))
    se_ax.format(xticks=[2, 5, 20], xminorticks=[10], xlabel='Return period (year)', xlim=(2, None), xlabelloc='top', xticklabelsize='med', xlabelpad=-1, xlabelsize='med',
                tickdir='in', ticklen=2, tickwidth=0.5, lw=0.5, ticklabelsize=8.5)