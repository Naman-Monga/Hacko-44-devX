# !pip install pycaret
# Pycaret Install kr lena and pandas too

import pandas as pd
dataset = pd.read_csv('/content/automobile.csv')
# path of the file ^

columnNameReg = 'price'
# column name of the column on which to apply & train liner regerssion models

# ---------------------------------------------------
# --------------------REGRESSION---------------------
# ---------------------------------------------------
from pycaret.regression import *

exp_reg101 = setup(data = dataset, target = columnNameReg, train_size = 0.7, categorical_features = None, categorical_imputation = 'constant', ordinal_features = None, high_cardinality_features = None, high_cardinality_method = 'frequency', numeric_features = None, numeric_imputation = 'mean', date_features = None, ignore_features = None, normalize = True, normalize_method = 'zscore', transformation = False, transformation_method = 'yeo-johnson', handle_unknown_categorical = True, unknown_categorical_method = 'least_frequent', pca = False, pca_method = 'linear', pca_components = None, ignore_low_variance = False, combine_rare_levels = False, rare_level_threshold = 0.10, bin_numeric_features = None, remove_outliers = False, outliers_threshold = 0.05, remove_multicollinearity = False, multicollinearity_threshold = 0.9, remove_perfect_collinearity = False, create_clusters = False, cluster_iter = 20, polynomial_features = False, polynomial_degree = 2, trigonometry_features = False, polynomial_threshold = 0.1, group_features = None, group_names = None, feature_selection = False, feature_selection_threshold = 0.8, feature_interaction = False, feature_ratio = False, interaction_threshold = 0.01, transform_target = False, transform_target_method = 'box-cox', data_split_shuffle = True, n_jobs = -1, html = True, session_id = None, log_experiment = False, experiment_name = None, log_plots = False, log_profile = False, log_data = False, silent=False, verbose = True, profile = False)
cmp = compare_models()
storecmp = pull()
#print(storecmp)
# storecmp is the pandas DataFrame containing all the R2, MAE, MSE, etc. values for 17 reg models

cmplist = storecmp.values.tolist()
print(cmplist)
# cmplist contains the above df in list format please view the dataframe once to get to know which value in the list are which ones

# ---------------------------------------------------
# --------------------CLASSIFICATION-----------------
# ---------------------------------------------------

columnNameClass = 'body-style'
from pycaret.classification import *
clf1 = setup(data = dataset, target = columnNameClass, train_size = 0.7, categorical_features = None, categorical_imputation = 'constant', ordinal_features = None, high_cardinality_features = None, high_cardinality_method = 'frequency', numeric_features = None, numeric_imputation = 'mean', date_features = None, ignore_features = None, normalize = True, normalize_method = 'zscore', transformation = False, transformation_method = 'yeo-johnson', handle_unknown_categorical = True, unknown_categorical_method = 'least_frequent', pca = False, pca_method = 'linear', pca_components = None, ignore_low_variance = False, combine_rare_levels = False, rare_level_threshold = 0.10, bin_numeric_features = None, remove_outliers = False, outliers_threshold = 0.05, remove_multicollinearity = False, multicollinearity_threshold = 0.9, remove_perfect_collinearity = False, create_clusters = False, cluster_iter = 20, polynomial_features = False, polynomial_degree = 2, trigonometry_features = False, polynomial_threshold = 0.1, group_features = None, group_names = None, feature_selection = False, feature_selection_threshold = 0.8, feature_interaction = False, feature_ratio = False, interaction_threshold = 0.01, fix_imbalance = False, fix_imbalance_method = None, data_split_shuffle = True, n_jobs = -1, html = True, session_id = None, log_experiment = False, experiment_name = None, log_plots = False, log_profile = False, log_data = False, silent=False, verbose=True, profile = False)
cmp2 = compare_models()
storecmp2 = pull()
#print(storecmp2)
# storecmp is the pandas DataFrame containing all the R2, MAE, MSE, etc. values for 14 classification models

cmplist2 = storecmp2.values.tolist()
print(cmplist2)
# cmplist contains the above df in list format please view the dataframe once to get to know which value in the list are which ones
