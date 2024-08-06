"""
This file is part of the accompanying code to our paper: Jiang, S., Tarasova, L., Yu, G., & Zscheischler, J. (2024). Compounding effects in flood drivers challenge estimates of extreme river floods. Science Advances, 10(13), eadl4005.

Copyright (c) 2024 Shijie Jiang. All rights reserved.
You should have received a copy of the MIT license along with the code. If not,
see <https://opensource.org/licenses/MIT>
"""

import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm
from sklearn.metrics import r2_score
from pyextremes import EVA
from scipy.signal import find_peaks

def get_station_data(fname, start_date='1981-01-01', end_date='2020-12-31'):
    dataset = pd.read_csv(fname, index_col=0, parse_dates=True)
    dataset = dataset.loc[start_date:end_date]
    dataset = dataset.astype(np.float32)  # Convert the dataset's data types to float32 for better memory efficiency
    
    return dataset

def get_wrapped_data(dataset, wrap_length=None):
    all_dates_list, data_x_list, data_y_list = [], [], []

    dataset_np = dataset.to_numpy()
    dateset_tm = dataset.index.strftime('%Y-%m-%d')

    for date_i in range(wrap_length, dataset_np.shape[0]):
        data_x = dataset_np[date_i-wrap_length:date_i+1, 0:-1]
        data_y = dataset_np[date_i, -1]
        date_value = dateset_tm[date_i]

        if (~np.isnan(data_y)) and (~np.isnan(data_x).any()):
            all_dates_list.append(date_value)
            data_x_list.append(np.concatenate([data_x[1:, 0], 
                                               np.mean(data_x[1:, 1], keepdims=True), 
                                               data_x[:1, 2], 
                                               data_x[:1, 3]], axis=0)), 
            data_y_list.append(data_y)

    return all_dates_list, data_x_list, data_y_list


def identify_all_peaks(Q, distance=None, k=0.75, valid_dates=None):
    candidate_peak_indices, _ = find_peaks(Q, distance=distance)
    #################################################################
    to_be_removed = []
    for i in range(len(candidate_peak_indices)-1):
        Q_peak_1 = Q.iloc[candidate_peak_indices[i]]
        Q_peak_2 = Q.iloc[candidate_peak_indices[i+1]]
        Q_low_between = Q.iloc[candidate_peak_indices[i]:candidate_peak_indices[i+1]].min()

        if Q_low_between   >= Q_peak_1 * k:
            to_be_removed.append(candidate_peak_indices[i])
        elif Q_low_between >= Q_peak_2 * k:
            to_be_removed.append(candidate_peak_indices[i+1])
        else:
            pass
    #################################################################    
    candidate_peak_indices = [i for i in candidate_peak_indices if i not in to_be_removed]
    
    Q_peaks = Q.iloc[candidate_peak_indices]
    peak_dates = Q_peaks.index.strftime('%Y-%m-%d')
    n = len(peak_dates)
    
    if valid_dates is not None:
        peak_dates = [peak_date for peak_date in peak_dates if peak_date in valid_dates]
        m = len(peak_dates)
        print(f"A total of {n} peaks are identified, and {m} are valid.")
    
    else:
        print(f"A total of {n} peaks are identified.")
    
    return peak_dates

def get_return_periods(extremes, plotting_position='weibull'):
    plotting_positions = {
        "ecdf": (0, 1),
        "hazen": (0.5, 0.5),
        "weibull": (0, 0),
        "tukey": (1 / 3, 1 / 3),
        "blom": (3 / 8, 3 / 8),
        "median": (0.3175, 0.3175),
        "cunnane": (0.4, 0.4),
        "gringorten": (0.44, 0.44),
        "beard": (0.31, 0.31),
    }
    ranks = (len(extremes) + 1 - sp.stats.rankdata(extremes.values, method="average"))
    alpha, beta = plotting_positions[plotting_position]

    exceedance_probability = (ranks - alpha) / (len(extremes) + 1 - alpha - beta)
    non_exceedance_probability = 1 - exceedance_probability

    return_periods = 1 / exceedance_probability

    extremes = extremes.copy(deep=True)

    return pd.DataFrame(
        data={
            extremes.name: extremes.values,
            "ranks": ranks,
            "exceedance probability": exceedance_probability,
            "non exceedance probability": non_exceedance_probability,
            "return period": return_periods,
        },
        index=extremes.index,
        dtype=np.float64,
    )

def get_estimate_largest(data, peak_series):
    peak_series = peak_series.copy()
    patial_peak_series = peak_series.drop(peak_series['fl'].idxmax())['fl'].sort_index().copy()
    patial_data = data[data.index.year.isin(peak_series.index.year)]
    
    #########
    # By default the distribution is selected automatically as best between 'genextreme' and 'gumbel_r'.
    # Best distribution is selected using the r2 metric.
    
    model_1 = EVA(patial_data['fl'])
    model_1.set_extremes(patial_peak_series, block_size='365.2425D')
    model_1.fit_model(model='MLE', distribution='genextreme')

    model_2 = EVA(patial_data['fl'])
    model_2.set_extremes(patial_peak_series, block_size='365.2425D')
    model_2.fit_model(model='MLE', distribution='gumbel_r')

    evaluate_df = get_return_periods(patial_peak_series)

    model_1_r2  = r2_score(evaluate_df['fl'], model_1.get_return_value(evaluate_df['return period'])[0])
    model_2_r2  = r2_score(evaluate_df['fl'], model_2.get_return_value(evaluate_df['return period'])[0])

    if model_1_r2 >= model_2_r2:
        best_model = model_1
    else:
        best_model = model_2
        
    #########
    real_value = peak_series.sort_values('fl').iloc[-1]['fl']
    esti_value = best_model.get_return_value(return_period=peak_series.sort_values('fl').iloc[-1]['return period'])[0]

    return real_value, esti_value


def get_explain_all(pd_peak_ex_ori):
    pd_peak_ex_all = pd_peak_ex_ori.copy().drop(pd_peak_ex_ori.columns, axis=1)
    
    pd_peak_ex_all['rr_shap'] = pd_peak_ex_ori[pd_peak_ex_ori.columns[pd_peak_ex_ori.columns.str.startswith('rr')]].sum(1)
    pd_peak_ex_all['tg_shap'] = pd_peak_ex_ori[pd_peak_ex_ori.columns[pd_peak_ex_ori.columns.str.startswith('tg')]].sum(1)
    pd_peak_ex_all['sm_shap'] = pd_peak_ex_ori[pd_peak_ex_ori.columns[pd_peak_ex_ori.columns.str.startswith('sm')]].sum(1)
    pd_peak_ex_all['sp_shap'] = pd_peak_ex_ori[pd_peak_ex_ori.columns[pd_peak_ex_ori.columns.str.startswith('sp')]].sum(1)
        
    return pd_peak_ex_all

def get_explain_ori(pd_peak_ex_ori, var_list):
    pd_peak_ex_ori2 = pd_peak_ex_ori.copy().drop(pd_peak_ex_ori.columns, axis=1)
    
    for index_1, var_1 in enumerate(var_list):
        for index_2, var_2 in enumerate(var_list):
            if index_1 < index_2:
                pd_peak_ex_ori2[f'{var_1}_x_{var_2}_shap'] = pd_peak_ex_ori[f'{var_1} x {var_2}'] + pd_peak_ex_ori[f'{var_2} x {var_1}']
            elif index_1 == index_2:
                pd_peak_ex_ori2[f'{var_1}_x_{var_2}_shap'] = pd_peak_ex_ori[f'{var_1} x {var_2}']
            else:
                pass
    
    return pd_peak_ex_ori2


def analyze_compounding_driver(pd_all_peak_eval, pd_is_tested, var_names, n_splits, n_repeats, quantile_v):
    pd_all_peak_ex_all_list     = []
    pd_all_peak_ex_all_imp_list = []
    all_threshold_list          = []

    for rep in tqdm(range(n_repeats)):
        pd_all_peak_ex_all_nn_list = []
        pd_all_peak_ex_all_imp_nn_list = []
        
        for j in range(n_splits):
            
            nn = rep * n_splits + j
            
            ex_ori_values_nn = np.vstack(pd_all_peak_eval[f'ex_int_{nn:03d}'])
            pd_all_peak_ex_ori_nn = pd.DataFrame(data=ex_ori_values_nn, 
                                        columns=[f'{var1} x {var2}' for var1 in var_names for var2 in var_names], 
                                        index=pd.to_datetime(pd_all_peak_eval.index))
            pd_all_peak_ex_all_nn = get_explain_all(pd_all_peak_ex_ori_nn)        
            pd_all_peak_ex_all_nn['y_true'] = pd_all_peak_eval['y_true']
            pd_all_peak_ex_all_nn = pd_all_peak_ex_all_nn[pd_is_tested[f'is_test_{nn:03d}']]
            pd_all_peak_ex_all_nn_list.append(pd_all_peak_ex_all_nn)
            
            pd_all_peak_ex_all_imp_nn = pd_all_peak_ex_all_nn[pd_all_peak_ex_all_nn.columns[pd_all_peak_ex_all_nn.columns.str.endswith('_shap')]]
            pd_all_peak_ex_all_imp_nn = pd_all_peak_ex_all_imp_nn[pd_is_tested[f'is_test_{nn:03d}']]
            pd_all_peak_ex_all_imp_nn_list.append(pd_all_peak_ex_all_imp_nn)
        
        ############################
        pd_all_peak_ex_all = pd.concat(pd_all_peak_ex_all_nn_list, axis=0)
        pd_all_peak_ex_all = pd_all_peak_ex_all.sort_index()
        
        values = pd_all_peak_ex_all[pd_all_peak_ex_all.columns[pd_all_peak_ex_all.columns.str.endswith('_shap')]].values
        all_threshold = np.quantile(values, quantile_v)
        ############################  
        pd_all_peak_ex_all_imp = pd.concat(pd_all_peak_ex_all_imp_nn_list, axis=0)
        pd_all_peak_ex_all_imp = pd_all_peak_ex_all_imp.sort_index()
        
        pd_all_peak_ex_all_imp = pd_all_peak_ex_all_imp > all_threshold
        pd_all_peak_ex_all_imp.columns = pd_all_peak_ex_all_imp.columns.str.replace('_shap', '_imp')
        pd_all_peak_ex_all_imp['rep'] = rep
        ############################
        pd_all_peak_ex_all_list.append(pd_all_peak_ex_all)
        pd_all_peak_ex_all_imp_list.append(pd_all_peak_ex_all_imp)
        all_threshold_list.append(all_threshold)
        
    pd_all_peak_ex_all_list     = pd.concat(pd_all_peak_ex_all_list)
    pd_all_peak_ex_all_imp_list = pd.concat(pd_all_peak_ex_all_imp_list)
    
    return pd_all_peak_ex_all_list, pd_all_peak_ex_all_imp_list, all_threshold_list


def analyze_flood_complexity(pd_all_peak_eval, pd_is_tested, var_names, n_splits, n_repeats, quantile_v):
    
    pd_all_peak_ex_ori_list     = []
    pd_all_peak_ex_ori_imp_list = []
    ori_threshold_list          = []

    for rep in tqdm(range(n_repeats)):
        pd_all_peak_ex_ori_nn_list = []
        pd_all_peak_ex_ori_imp_nn_list = []
        
        for j in range(n_splits):
            
            nn = rep * n_splits + j
            
            ex_ori_values_nn = np.vstack(pd_all_peak_eval[f'ex_int_{nn:03d}'])
            pd_all_peak_ex_ori_nn = pd.DataFrame(data=ex_ori_values_nn, 
                                        columns=[f'{var1} x {var2}' for var1 in var_names for var2 in var_names], 
                                        index=pd.to_datetime(pd_all_peak_eval.index))
            pd_all_peak_ex_ori_nn = get_explain_ori(pd_all_peak_ex_ori_nn, var_list=var_names)

            pd_all_peak_ex_ori_nn['y_true'] = pd_all_peak_eval['y_true']
            pd_all_peak_ex_ori_nn['exceedance probability'] = pd_all_peak_eval['exceedance probability']
            pd_all_peak_ex_ori_nn = pd_all_peak_ex_ori_nn[pd_is_tested[f'is_test_{nn:03d}']]
            pd_all_peak_ex_ori_nn_list.append(pd_all_peak_ex_ori_nn)
            
            pd_all_peak_ex_ori_imp_nn = pd_all_peak_ex_ori_nn[pd_all_peak_ex_ori_nn.columns[pd_all_peak_ex_ori_nn.columns.str.endswith('_shap')]]
            pd_all_peak_ex_ori_imp_nn = pd_all_peak_ex_ori_imp_nn[pd_is_tested[f'is_test_{nn:03d}']]
            pd_all_peak_ex_ori_imp_nn_list.append(pd_all_peak_ex_ori_imp_nn)
        ############################
        pd_all_peak_ex_ori = pd.concat(pd_all_peak_ex_ori_nn_list, axis=0)
        pd_all_peak_ex_ori = pd_all_peak_ex_ori.sort_index()
        values = pd_all_peak_ex_ori[pd_all_peak_ex_ori.columns[pd_all_peak_ex_ori.columns.str.endswith('_shap')]].values
        ori_threshold = np.quantile(values[values > 0], quantile_v)
        ############################  
        pd_all_peak_ex_ori_imp = pd.concat(pd_all_peak_ex_ori_imp_nn_list, axis=0)
        pd_all_peak_ex_ori_imp = pd_all_peak_ex_ori_imp.sort_index()

        pd_all_peak_ex_ori_imp = pd_all_peak_ex_ori_imp > ori_threshold

        pd_all_peak_ex_ori_imp.columns = pd_all_peak_ex_ori_imp.columns.str.replace('_shap', '_imp')
        pd_all_peak_ex_ori_imp = pd_all_peak_ex_ori_imp.merge(pd_all_peak_ex_ori[['y_true', 'exceedance probability']], left_index=True, right_index=True)
        pd_all_peak_ex_ori_imp['rep'] = rep
        
        ############################
        pd_all_peak_ex_ori_list.append(pd_all_peak_ex_ori)
        pd_all_peak_ex_ori_imp_list.append(pd_all_peak_ex_ori_imp)
        ori_threshold_list.append(ori_threshold)
        
    pd_all_peak_ex_ori_list     = pd.concat(pd_all_peak_ex_ori_list)
    pd_all_peak_ex_ori_imp_list = pd.concat(pd_all_peak_ex_ori_imp_list)

    return pd_all_peak_ex_ori_list, pd_all_peak_ex_ori_imp_list, ori_threshold_list