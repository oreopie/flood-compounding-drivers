"""
This file is part of the accompanying code to our paper: Jiang, S., Tarasova, L., Yu, G.,& Zscheischler, J., The importance of compounding drivers for large river floods.

Copyright (c) 2023 Shijie Jiang. All rights reserved.
You should have received a copy of the MIT license along with the code. If not,
see <https://opensource.org/licenses/MIT>
"""
##################################################################
import os
import shap
import warnings
import lightgbm
import argparse
import pickle

import numpy as np
import scipy as sp
import pandas as pd

from sklearn import model_selection
from tqdm import tqdm
from sklearn.metrics import r2_score

from libs import utils, plots
##################################################################
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--input_path", help="The path to the basin data for analysis", required=True, type=str)
argParser.add_argument("-s", "--basin_size", help="The size of the basin (in km2)", required=True, type=float)
argParser.add_argument("-o", "--output_dir", help="The directory to save the output", required=True, type=str)
argParser.add_argument("-l", "--lag_days", help="The day length used to consider the inputs (default=7)", default=7, type=int)
argParser.add_argument("-q", "--q_for_therds", help="The quantile used to determine the thresholds (default=0.8)", default=0.8, type=float)
argParser.add_argument("-n", "--n_splits", help="n-Fold Cross-Validation (default=5)", default=5, type=int)
argParser.add_argument("-m", "--n_repeats", help="The number of replicates (default=100)", default=100, type=int)
args = argParser.parse_args()

input_path = args.input_path
basin_size = args.basin_size
output_dir = args.output_dir
lag_days   = args.lag_days
quantile_v = args.q_for_therds
n_splits   = args.n_splits
n_repeats  = args.n_repeats

lag_days_for_peak = int(5 + np.log(basin_size / 2.59))  # Used for detecting identifiable peaks
var_names = [f'rr_{i}' for i in range(lag_days-1, -1, -1)] + ['tg_avg'] + [f'sm_{lag_days}'] + [f'sp_{lag_days}']
##################################################################
hydrodata    = utils.get_station_data('data/sample.csv')
hydrodata    = hydrodata[['rain', 'tg', 'sm', 'sp','fl']]

print("\nData loaded successfully!")
print(hydrodata)
##################################################################
# Here we wrap the data to input-output samples
# the input has a shape of [n, 10], where n is the number of samples, 10 features include:
# ['rr_6', 'rr_5', 'rr_4', 'rr_3', 'rr_2', 'rr_1', 'rr_0', 'tg_avg', 'sm_7', 'sp_7'] (var_names)
##   rr -- rainfall
##   tg -- temperature
##   sm -- soil moisture
##   sp -- snowpack
##   the number represents the number of days before a discharge

all_dates_list, data_x_list, data_y_list = utils.get_wrapped_data(hydrodata, wrap_length=lag_days)
all_dates_list = pd.DatetimeIndex(all_dates_list)

var_names = [f'rr_{i}' for i in range(lag_days-1, -1, -1)] + ['tg_avg'] + [f'sm_{lag_days}'] + [f'sp_{lag_days}']

all_peak_dates = utils.identify_all_peaks(Q=hydrodata['fl'], 
                                          distance=lag_days_for_peak, 
                                          k=0.75, 
                                          valid_dates=all_dates_list, # we only consider the peaks that have valid inputs
                                         ) 
all_peak_dates = pd.DatetimeIndex(all_peak_dates)                     # the dates of all identifiable peaks

peak_series    = hydrodata.loc[all_peak_dates, 'fl']                  # the pandas series of all identifiable peaks
##################################################################
# am_peak_dates: the dates of the annual maximum (AM) flood events 
# note: we only take into account the peak for which the year has more than 200 observations
# am_series: the pandas dataframe of all AM flood events (contains empirical probability and return period)

am_peak_dates  = peak_series.groupby(peak_series.index.year).idxmax()
hydrodata_fl_enough_length = hydrodata['fl'].groupby(hydrodata.index.year).count() >= 200

am_peak_dates  = am_peak_dates[hydrodata_fl_enough_length]
am_series      = utils.get_return_periods(hydrodata.loc[am_peak_dates, 'fl'], plotting_position='weibull')

real_value, esti_value = utils.get_estimate_largest(hydrodata, am_series)
estimation_error = (esti_value - real_value) / real_value
print('The estimation error of the largest observed flood based on all annual flood events except the largest one:')
print(f'{estimation_error:0.3%}')
##################################################################
# Prepare pandas dataframe to train the model and store the interpretation results

pd_data = pd.DataFrame(data={'x_values': data_x_list,
                             'y_true': data_y_list,
                             'all_peak': False,
                             'am_peak': False},
                       index=pd.to_datetime(all_dates_list))
pd_data.loc[all_peak_dates, 'all_peak'] = True
pd_data.loc[am_peak_dates,  'am_peak'] = True

pd_data_run           = pd.DataFrame(index=pd_data.index, data=np.vstack(pd_data['x_values']), columns=var_names)
pd_data_run['y_true'] = pd_data['y_true']

pd_data_eval          = pd_data.copy()

pd_all_peak_eval      = pd_data_eval[pd_data_eval['all_peak']]
pd_all_peak_eval      = pd_all_peak_eval.merge(am_series[['ranks', 
                                                          'exceedance probability', 
                                                          'non exceedance probability',
                                                          'return period']], 
                                               left_index=True, right_index=True, how='left')

##################################################################
# The following is for model training
# The model will be trained by using LightGBM with Repeated n-fold Cross-Validation.
# Hyperparameters are determined by randomized search.
##################################################################
r2_df      = pd.DataFrame(columns=['r2_train', 'r2_test'])
r2_df.index.name = 'n'

model_list = []

kf = model_selection.RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
nn  = 0

pd_is_tested_nn_list = []
print("="*30)
print(f"The model started training and Repeated {n_splits}-Fold Cross-Validation is used (n_repeats={n_repeats}). It may take a while.")
#########################################
# We use monotone_constraints and interaction_constraints (see Methods in the paper)
monotone_constraints = [1 for i in range(lag_days)] + [0, 1, 1]

interactions_all = [[i, j] for i in [k for k in range(lag_days+3)] for j in [k for k in range(lag_days+3)] if i <= j]
interactions_rm1 = [[i, j] for i in [k for k in range(lag_days)] for j in [k for k in range(lag_days, lag_days+1)] if i < j]
interaction_constraints = [s for s in interactions_all if s not in interactions_rm1]
#########################################
for train_index, test_index in tqdm(kf.split(np.arange(len(all_peak_dates)))):
    train_dates = all_peak_dates[train_index]
    test_dates  = all_peak_dates[test_index]
    #########################################
    train_x     = pd_data_run.loc[train_dates, pd_data_run.columns!='y_true'].values
    train_y     = pd_data_run.loc[train_dates, 'y_true'].values
    test_x      = pd_data_run.loc[test_dates, pd_data_run.columns!='y_true'].values
    test_y      = pd_data_run.loc[test_dates, 'y_true'].values
    ########## 
    pd_is_tested_nn = pd.DataFrame(index=all_peak_dates, columns=[f'is_test_{nn:03d}'], data=False)
    pd_is_tested_nn.loc[test_dates] = True
    pd_is_tested_nn_list.append(pd_is_tested_nn)
    #########################################
    candidate_params = {
        'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05],
        'n_estimators': [50, 100, 150, 200],
        'subsample': [0.3, 0.5, 0.7],
        'colsample_bytree': [0.3, 0.5, 0.7],
        'max_bin': [8, 16, 24, 32, 64],
        'monotone_constraints': [monotone_constraints],
        'interaction_constraints': [interaction_constraints],
        'min_child_samples': [3, 5, 7]
    }
    grid = model_selection.RandomizedSearchCV(lightgbm.LGBMRegressor(), 
                                              candidate_params, 
                                              n_iter=64, 
                                              scoring='r2', 
                                              cv=5, 
                                              n_jobs=-2, 
                                              random_state=42 * nn)
    grid.fit(train_x, train_y)
    
    clf_model   = lightgbm.LGBMRegressor(**grid.best_params_)
    clf_model.fit(train_x, train_y, 
                  callbacks=[lightgbm.early_stopping(stopping_rounds=8, verbose=0)],
                  eval_set=[(train_x, train_y)], verbose=0
                 )
    model_list.append(clf_model)
    #########################################
    train_pred = clf_model.predict(train_x)
    test_pred  = clf_model.predict(test_x)
    train_r2   = r2_score(y_true=train_y, y_pred=train_pred)
    test_r2    = r2_score(y_true=test_y,  y_pred=test_pred)
    
    r2_df      = r2_df.append({'r2_train': train_r2, 'r2_test': test_r2}, ignore_index=True)
    #########################################
    nn = nn + 1
    
pd_is_tested = pd.concat(pd_is_tested_nn_list, axis=1) # used to record which samples were for testing in each replicate

model_path   = os.path.join(output_dir, 'sample_trained_models.bin')
with open(model_path, 'wb') as handle:
    pickle.dump([model_list, pd_is_tested], handle)

print(f"Training finished, models saved to {model_path}")
print(f"mean_r2_train: {np.mean(r2_df['r2_train']):.3f}")  
print(f"mean_r2_test:  {np.mean(r2_df['r2_test']):.3f}")

##################################################################
# The following is for model interpretation
# We use SHAP interaction values to explain the trained models.
##################################################################
print("="*30)
print(f"We use SHAP interaction values to explain the trained models")

intetpret_model_list = model_list.copy()

pd_all_peak_eval_list = []

for n, intetpret_model in tqdm(enumerate(intetpret_model_list)):
    pd_data_expalin = pd_data_eval[pd_data_eval['all_peak']]
    pd_data_expalin = pd_data_expalin.sort_index()
    pd_data_expalin = pd.DataFrame(index=pd_data_expalin.index, 
                                   data=np.vstack(pd_data_expalin['x_values']), 
                                   columns=var_names)
    ########################################################
    pd_all_peak_eval_n = pd.DataFrame(index=pd_data_expalin.index)
    ########################################################
    peak_x = pd_data_expalin.values
    peak_y = pd_data_eval[pd_data_eval['all_peak']].sort_index()['y_true'].values
    peak_pred = intetpret_model.predict(peak_x)
    ############################
    explainer_int           = shap.TreeExplainer(intetpret_model, 
                                                 feature_names=var_names)
    
    shap_interaction_values = explainer_int.shap_interaction_values(X=peak_x, y=peak_y)
    shap_interaction_values = shap_interaction_values.reshape([-1, len(var_names)*len(var_names)])
    ########################################################
    pd_all_peak_eval_n[f'y_pred_{n:03d}'] = peak_pred                         # the predicted output 
    pd_all_peak_eval_n[f'y_bar_{n:03d}']  = explainer_int.expected_value      # the expected value of the model output 
    pd_all_peak_eval_n[f'ex_int_{n:03d}'] = shap_interaction_values.tolist()  # the SHAP interaction values (100 elements because of 10 features)
    pd_all_peak_eval_list.append(pd_all_peak_eval_n)
    
pd_all_peak_eval = pd_all_peak_eval.merge(pd.concat(pd_all_peak_eval_list, axis=1), left_index=True, right_index=True)
pd_all_peak_eval.index.name = 'date'

explain_path = os.path.join(output_dir, 'sample_model_explanations.p')
pd_all_peak_eval.to_pickle(explain_path)

print(f"Interpretation finished, interpretation results saved to {explain_path}")
##################################################################
# The following is for the results of compounding drivers
# We use the aggregated SHAP interaction values to determine which main drivers are associated with each of individual flood events.
##################################################################
# pd_all_peak_ex_all: the aggregated SHAP interaction values of the four drivers for each identifiable peak
# pd_all_peak_ex_all_imp: whether the driver is a main driver for each identifiable peak
# pd_am_peak_ex_all_imp: whether the driver is a main driver for each AM flood peak
print("="*30)
print(f"We use aggregated SHAP interaction values to identify multi-driver floods")
pd_all_peak_ex_all, pd_all_peak_ex_all_imp, all_thresholds = utils.analyze_compounding_driver(pd_all_peak_eval, 
                                                                        pd_is_tested,
                                                                        var_names,
                                                                        n_splits,
                                                                        n_repeats,
                                                                        quantile_v=quantile_v # using the 80th percentile as thred
                                                                        )
pd_am_peak_ex_all_imp = pd_all_peak_ex_all_imp.loc[am_peak_dates]

# An event associated with at least two main drivers is regarded as a multi-driver event
pd_am_peak_ex_all_imp['mu_imp'] = pd_am_peak_ex_all_imp['rr_imp'].astype(int) + \
                                  pd_am_peak_ex_all_imp['tg_imp'].astype(int) + \
                                  pd_am_peak_ex_all_imp['sm_imp'].astype(int) + \
                                  pd_am_peak_ex_all_imp['sp_imp'].astype(int)
pd_am_peak_ex_all_imp['mu_imp'] = pd_am_peak_ex_all_imp['mu_imp'] >= 2

# pd_am_peak_sum: the number of exceedances of the threshold for each variable/combination
pd_am_peak_sum = pd_am_peak_ex_all_imp.groupby(pd_am_peak_ex_all_imp.index).sum()[['rr_imp',
                                                                                   'tg_imp',
                                                                                   'sm_imp',
                                                                                   'sp_imp', 
                                                                                   'mu_imp']].add_suffix('_sum')
print('The following table shows the number of exceedances of the threshold for each variable/combination')
###########
for k_ex in range(1, n_repeats+1):
    pvalue = sp.stats.binomtest(k_ex, n=n_repeats, p=0.5, alternative='greater').pvalue
    if pvalue < 0.01:
        break
        
# With {n_repeats} replicates and a significance level of 0.01, a driver or a combination
# with at least {k_ex} exceedances of the respective thresholds is considered to be
# significantly associated with the corresponding flood event.

pd_am_peak_sig = (pd_am_peak_sum >= k_ex).add_suffix('_sig')  

pd_am_peak_sig_num = pd_am_peak_sig.sum().add_suffix('_num')
pd_am_peak_sig_num_prop = pd_am_peak_sig_num.div(len(pd_am_peak_sig), axis = 'rows')
pd_am_peak_sig_num_prop.index = pd_am_peak_sig_num_prop.index.str.replace("_num", "_prop")
###########
print(f"The proportion of recent rainfall as a main driver of AM floods:")
print(f"{pd_am_peak_sig_num_prop['rr_imp_sum_sig_prop']:0.3f}")
print(f"The proportion of recent temperature as a main driver of AM floods:")
print(f"{pd_am_peak_sig_num_prop['tg_imp_sum_sig_prop']:0.3f}")
print(f"The proportion of soil moisturel as a main driver of AM floods:")
print(f"{pd_am_peak_sig_num_prop['sm_imp_sum_sig_prop']:0.3f}")
print(f"The proportion of snowpack as a main driver of AM floods:")
print(f"{pd_am_peak_sig_num_prop['sp_imp_sum_sig_prop']:0.3f}")
print(f"The proportion of multi-driver floods:")
print(f"{pd_am_peak_sig_num_prop['mu_imp_sum_sig_prop']:0.3f}")
###########
pd_am_peak_sum['multi_driver']      = pd_am_peak_sum['mu_imp_sum'] >= k_ex
pd_am_peak_sum['single_driver']     = pd_am_peak_sum['mu_imp_sum'] <= (n_repeats - k_ex)

pd_mag = pd.merge(pd_am_peak_sum[['multi_driver', 'single_driver']], pd_data['y_true'], 
                  left_index=True, right_index=True, how='left')

mag_ratio = pd_mag[pd_mag['multi_driver']]['y_true'].mean() / pd_mag[pd_mag['single_driver']]['y_true'].mean()
mag_ttest = sp.stats.ttest_ind(pd_mag[pd_mag['multi_driver']]['y_true'], pd_mag[pd_mag['single_driver']]['y_true'], alternative='greater').pvalue
print("="*30)
print(f'Magnitude ratio of multi-driver floods to single-driver floods:\n{mag_ratio:0.3f}')
print(f'T-test p-value for the mean magnitude difference:\n{mag_ttest:0.2g}')

##################################################################
# The following is for the results of flood complexity
##################################################################
print("="*30)
print(f"We use the original SHAP interaction values to calculate the flood complexity")
# pd_all_peak_ex_ori: the original SHAP interaction values of the 55 interactions for each identifiable peak
# pd_all_peak_ex_ori_imp: whether the interaction is a main interaction for each identifiable peak
# pd_am_peak_ex_ori_imp: whether the interaction is a main interaction for each AM flood peak

pd_all_peak_ex_ori, pd_all_peak_ex_ori_imp, ori_thresholds = utils.analyze_flood_complexity(pd_all_peak_eval,
                                                                        pd_is_tested,
                                                                        var_names,
                                                                        n_splits,
                                                                        n_repeats,
                                                                        quantile_v=quantile_v # using the 80th percentile
                                                                        )

pd_am_peak_ex_ori_imp = pd_all_peak_ex_ori_imp.loc[am_peak_dates]

pd_am_peak_ex_ori_imp['num_interaction']  = pd_am_peak_ex_ori_imp[pd_am_peak_ex_ori_imp.columns[pd_am_peak_ex_ori_imp.columns.str.endswith('_imp')]].astype(int).sum(1)

# the richness of interactions (see Extended Data Fig. 4)
pd_am_peak_ex_ori_imp['prop_interaction'] = pd_am_peak_ex_ori_imp['num_interaction'] / 48 * 100 

# slope of the fitted line
com_slope_s_complex = pd_am_peak_ex_ori_imp.groupby('rep').apply(lambda x: sp.stats.linregress(100 - x['exceedance probability']*100, x['prop_interaction'], alternative='greater')[0])

# intercept of the fitted line
com_slope_i_complex = pd_am_peak_ex_ori_imp.groupby('rep').apply(lambda x: sp.stats.linregress(100 - x['exceedance probability']*100, x['prop_interaction'], alternative='greater')[1])

# pvalue of the fitted line
com_slope_p_complex = pd_am_peak_ex_ori_imp.groupby('rep').apply(lambda x: sp.stats.linregress(100 - x['exceedance probability']*100, x['prop_interaction'], alternative='greater')[3]) 

com_linear_slope_median = np.median(com_slope_s_complex)
combined_pvalues        = sp.stats.combine_pvalues(com_slope_p_complex, method='fisher')[1]

print(f'Flood complexity:\n{com_linear_slope_median:0.3f}')
print(f'Combined p-value for the flood complexity:\n{combined_pvalues:0.2g}')

##################################################################
# Save all results
##################################################################
result = pd.DataFrame(data={'prop_rr': pd_am_peak_sig_num_prop['rr_imp_sum_sig_prop'].round(3),
                            'prop_tg': pd_am_peak_sig_num_prop['tg_imp_sum_sig_prop'].round(3),
                            'prop_sm': pd_am_peak_sig_num_prop['sm_imp_sum_sig_prop'].round(3),
                            'prop_sp': pd_am_peak_sig_num_prop['sp_imp_sum_sig_prop'].round(3),
                            'prop_mu': pd_am_peak_sig_num_prop['mu_imp_sum_sig_prop'].round(3),
                            'mag_ratio': np.round(mag_ratio, 3),
                            'mag_ttest_p': np.round(mag_ttest, 3),
                            'flood_com': np.round(com_linear_slope_median, 3),
                            'flood_com_p': np.round(combined_pvalues, 3),
                            'est_err': np.round(estimation_error*100, 3)}, index=[0])

result_path = os.path.join(output_dir, 'sample_result.csv')
result.to_csv(result_path, index=False)
print("="*30)
print(result)
print(f"The result is saved to {result_path}")