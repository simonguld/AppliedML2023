# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import pickle
import os, sys
import numpy as np
import seaborn as sns
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #  {'0', '1', '2', '3'} = {Show all messages, remove info, remove info and warnings, remove all messages}
import tensorflow as tf
from tensorflow import keras
from keras import models, layers, regularizers
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.cluster import MiniBatchKMeans, KMeans, Birch, BisectingKMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, roc_curve, auc, log_loss, r2_score, silhouette_score
from sklearn.neural_network import MLPClassifier
 
#from category_encoders import LeaveOneOutEncoder, TargetEncoder
import xgboost as xgb
from eli5.sklearn import PermutationImportance
import optuna
from optuna.samplers import TPESampler
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import MedianPruner
import lightgbm as lgb

sys.path.append('Appstat2022\\External_Functions')
sys.path.append('AdvAppStat')
from statistics_helper_functions import calc_ROC, calc_fisher_discrimminant, calc_ROC_AUC
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure

## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


pd.options.mode.chained_assignment = None



## Set plotting style and print options
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster

d = {'lines.linewidth': 2, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12,\
     'legend.fontsize': 15, 'font.family': 'serif', 'figure.figsize': (9,6)}
d_colors = {'axes.prop_cycle': cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'olivedrab',\
         'black', 'red', 'cyan', 'yellow', 'khaki','lightblue'])}
rcParams.update(d)
rcParams.update(d_colors)
np.set_printoptions(precision = 5, suppress=1e-10)


### FUNCTIONS ----------------------------------------------------------------------------------

def load_data(name):
    with h5py.File(f'{name}.h5', 'r') as f:
        filename = name.split('/')[1]
        return pd.DataFrame(f[filename][:], dtype=np.float64)

def evaluate_classification_results(estimator, X_train, X_val, y_train, y_val, method_name, plot = False):
         ## Step 1: Visualize to ensure overtraining has not ocurred
            # Evaluate:
            train_pred = estimator.predict_proba(X_train)[:,1]
            val_pred = estimator.predict_proba(X_val)[:,1]
        
            fpr_train, tpr_train, _ = roc_curve(y_train, train_pred)                  # False/True Positive Rate for our model
            fpr_val, tpr_val, _ = roc_curve(y_val, val_pred)  # False/True Positive Rate for Aleph NNbjet

            # We can now calculate the AUC scores of these ROC-curves:
            auc_score_train = auc(fpr_train,tpr_train)                       
            auc_score_val = auc(fpr_val, tpr_val)  

            log_loss_val = log_loss(y_val, val_pred)
            log_loss_train = log_loss(y_train, train_pred)

            benchmark_stats_val = np.array([log_loss_val, estimator.score(X_val,y_val), auc_score_val ])
            benchmark_stats_train = np.array([log_loss_train, estimator.score(X_train,y_train), auc_score_train ])
            
            print("binary/acc/AUC val: ", benchmark_stats_val)
            print("binary/acc/AUC train: ", benchmark_stats_train)

            if plot:
                fig, ax = plt.subplots()
                ax.plot(fpr_train, tpr_train, label=f'Train (AUC = {auc_score_train:5.3f})')
                ax.plot(fpr_val, tpr_val, label = f'Val (AUC = {auc_score_val:5.3f})')
                ax.set(title=f'{method_name} ROC for train and val data', xlabel = 'FPR', ylabel = 'TPR')
                ax.legend()
     
            return benchmark_stats_train, benchmark_stats_val
    
def evaluate_predictions(y_pred, y_true, label = 'train', quantiles = None, verbose = 1):
            if verbose:
                print(f'\nFor {label} data:')

            f_list = [r2_score, metrics.mean_absolute_error, metrics.mean_absolute_percentage_error] #, mean_squared_log_error]
            name_list = ['r2', 'MAE', 'MAPE']
            res_list = [None] * len(f_list)
            for i, f in enumerate(f_list):
                if name_list[i] == 'MSQLE':
                    res_list[i] = f(y_true, np.abs(y_pred))
                else:
                    res_list[i] = f(y_true, y_pred)
                if verbose:
                    print(f'{name_list[i]}: ', res_list[i])

            if 0:
                rel_err = np.abs(y_pred - y_true) / y_true
                if quantiles is None:
                    Nstatistics = 7
                    quantiles = [.05, .25, .50, .75, .95, .99]
                    statistics = [None] * Nstatistics
                    statistics[0] = rel_err.mean()
                    statistics[1:] = np.quantile(rel_err, quantiles)
                else:
                    Nstatistics = len(quantiles)
                    statistics = np.quantile(rel_err, quantiles)
                if verbose:
                    print(f"mean and {quantiles} quantiles of rel error: ", statistics)

            return res_list
  
def visualize_regression(y_pred_train, y_pred_val, y_train, y_val, model_name):
                ## Visualize
                fig, ax = plt.subplots(ncols=2)
                ax = ax.flatten()
                fig.suptitle(f'True vs predicted for {model_name}')
                ax[0].plot(y_val, y_pred_val, '.', markersize = 1.5, alpha = .6)
                ax[0].set(xlabel = 'True energy',\
                        ylabel = 'Predicted energy', title = 'Val', ylim=(0,400_000))
                ax[1].plot(y_train, y_pred_train, '.', markersize = 1.5, alpha = .6)
                ax[1].set(xlabel = 'True energy', ylabel = 'Predicted energy', title = 'Train', ylim=(0,400_000))
                ax[1].yaxis.set_visible(False)

                # Plot rel. error distribution 
                rel_err_val = (y_pred_val - y_val) / y_val
                rel_err_train = (y_pred_train - y_train) / y_train


                fig2, ax2 = plt.subplots()
                range = (min(np.min(y_pred_val), np.min(y_val)), max(np.max(y_pred_val), np.max(y_val)))
                ax2.set(title = 'Energy distribution for true and predicted values')
                counts, _, _ = ax2.hist(y_pred_val, range = range,  bins = 600, histtype='stepfilled', alpha=.7, label = 'Pred val')
                counts_true,_,_ = ax2.hist(y_val, range = range, bins = 600, histtype= 'step', label='True')
                print(counts.sum(), counts_true.sum())
                ax2.legend()


def clustering_accuracy(X_train, y_train, clustering_labels, Nlabels = None):

    if Nlabels is None:
        Nlabels = len(np.unique(clustering_labels))

    Nsamples = X_train.shape[0]
    acc_arr = np.empty(Nlabels)
    weight_arr = np.empty_like(acc_arr)

    for label in np.arange(Nlabels):
        mask = (clustering_labels == label)
        Ncluster = mask.sum()
        Nelectron = np.sum((y_train[mask]))
        if Ncluster == 0:
            acc_arr[label] = 0
        else:
            acc_arr[label] = max(Nelectron / Ncluster, 1 - Nelectron / Ncluster)
        weight_arr[label] = Ncluster / Nsamples

    acc = (weight_arr * acc_arr).sum()

    return acc, acc_arr, weight_arr


def objective(trial, method, param_wrapper, scoring, X_train, y_train):

    params = param_wrapper(trial)
    clf = method(**params)

    cv_results = cross_validate(clf, X_train, y_train, scoring=scoring, cv=5) #, fit_params=xgb_params)
    print(cv_results['test_score'])
    return cv_results['test_score'].mean()
            
def xgb_classifier(X_train, X_val, y_train, y_val, test_features):
        feature_importance_evaluation, hyperoptimization, train_best_model = False, False, False

        if feature_importance_evaluation:
            xgb_params = dict(objective='binary:logistic', learning_rate=0.3, min_split_loss = 0, min_child_weight = 1, subsample = 1,\
                            colsample_bytree = 1, colsample_bylevel = 1,
                                        reg_lambda = 1, reg_alpha = 0, max_depth=4, eval_metric='logloss', n_estimators=50,)

            xgb_clf = xgb.XGBClassifier(**xgb_params)
            xgb_clf.fit(X_train, y_train)


            ## Step 1: Visualize to ensure overtraining has not ocurred
            # Evaluate:
            train_pred = xgb_clf.predict_proba(X_train)[:,1]
            val_pred = xgb_clf.predict_proba(X_val)[:,1]
        
            fpr_train, tpr_train, _ = roc_curve(y_train, train_pred)                  # False/True Positive Rate for our model
            fpr_val, tpr_val, _ = roc_curve(y_val, val_pred)  # False/True Positive Rate for Aleph NNbjet

            # We can now calculate the AUC scores of these ROC-curves:
            auc_score_train = auc(fpr_train,tpr_train)                       
            auc_score_val = auc(fpr_val, tpr_val)  

            print("Val acc. before selecting best 15 params: ", xgb_clf.score(X_val,y_val))
            print("Train acc. before selecting best 15: ", xgb_clf.score(X_train,y_train))

            
            ## Step 2: Extract the 15 most important features
            #good_features_xgb_indices = np.argwhere(xgb_clf.feature_importances_ > 0.01).flatten()
            good_features_xgb_indices = np.argsort(-xgb_clf.feature_importances_)[:15]
            good_features_xgb = X_train.columns[good_features_xgb_indices]
            print(len(good_features_xgb))

            # Save good features for xgb
            np.savetxt('Classification_SimonGuldager_XGBoost_VariableList.txt', np.array(good_features_xgb, dtype='str'), delimiter=',',fmt="%s")

            # Fit again to ensure that performance is not hurt
            xgb_clf.fit(X_train[good_features_xgb], y_train)
            print(xgb_clf.n_features_in_)
            print(xgb_clf.feature_importances_)

            print("Val acc after: ",xgb_clf.score(X_val[good_features_xgb],y_val))
            print("Train acc after: ", xgb_clf.score(X_train[good_features_xgb],y_train))     

            log_loss_val = log_loss(y_val, val_pred)
            print("log_loss_val: ", log_loss_val)

        # print([log_loss_val, xgb_clf.score(X_val[good_features_xgb],y_val), auc_score_val ])
            benchmark_stats = np.array([log_loss_val, xgb_clf.score(X_val[good_features_xgb],y_val), auc_score_val ])
            np.savetxt('benchmark_performance_xgb.txt', benchmark_stats)
        elif hyperoptimization:

            
            good_features_xgb = list(np.loadtxt('Classification_SimonGuldager_XGBoost_VariableList.txt', dtype = "str"))

            def objective(trial, method, scoring):

                # Constant params
                xgb_params = dict(objective = 'binary:logistic', min_split_loss = 0, min_child_weight = 1, subsample = 1,\
                                    colsample_bytree = 1, colsample_bylevel = 1, reg_lambda = 1, reg_alpha = 0, 
                                    eval_metric='logloss')
                # Parameters to vary
                xgb_params_var = {#'subsample': trial.suggest_float('subsample', 0.6,1),
                            'n_estimators': trial.suggest_int('n_estimators',50,100),
                            'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.4),
                            'max_depth': trial.suggest_int('max_depth', 4,6)}
                xgb_params.update(xgb_params_var)

                xgb_clf = method(**xgb_params)

                cv_results = cross_validate(xgb_clf, X_train[good_features_xgb], y_train, scoring=scoring, cv=5) #, fit_params=xgb_params)
                print(cv_results['test_score'])
                return cv_results['test_score'].mean()
            
            study = optuna.create_study(direction="maximize",sampler=TPESampler(),pruner=MedianPruner(n_warmup_steps=50))
            xgb_clf = xgb.XGBClassifier()
            study.optimize(lambda trial: objective(trial, xgb.XGBClassifier, scoring='neg_log_loss'), n_trials=25, show_progress_bar=False)
        
            print(study.best_trial.params)
            print(study.best_trial.values)

            xgb_params = dict(objective = 'binary:logistic', min_split_loss = 0, min_child_weight = 1, subsample = 1,\
                                    colsample_bytree = 1, colsample_bylevel = 1, reg_lambda = 1, reg_alpha = 0, 
                                    eval_metric='logloss', early_stopping_rounds=100)
            
            xgb_params.update(study.best_trial.params)
            print("Optimized parameters", xgb_params)
            ## Fit using the best parameters and evaluate results
            xgb_clf = xgb.XGBClassifier(**xgb_params)
            xgb_clf.fit(X_train[good_features_xgb], y_train, eval_set = [(X_val[good_features_xgb], y_val)])

            print("Best score:", xgb_clf.best_score)

            train_pred = xgb_clf.predict_proba(X_train[good_features_xgb])[:,1]
            val_pred = xgb_clf.predict_proba(X_val[good_features_xgb])[:,1]
    
            fpr_train, tpr_train, _ = roc_curve(y_train, train_pred)                  # False/True Positive Rate for our model
            fpr_val, tpr_val, _ = roc_curve(y_val, val_pred)  # False/True Positive Rate for Aleph NNbjet

            # We can now calculate the AUC scores of these ROC-curves:
            auc_score_train = auc(fpr_train,tpr_train)                       
            auc_score_val = auc(fpr_val, tpr_val)  

            print("Val acc. after param optimization", xgb_clf.score(X_val[good_features_xgb],y_val))
            print("Train acc. after param optimization ", xgb_clf.score(X_train[good_features_xgb],y_train))

            benchmark_stats = np.loadtxt('benchmark_performance_xgb.txt')
            print("Benchmark logloss/acc/AUC: ", benchmark_stats)


            ## save beset model
            xgb_clf.save_model('xgb_classifier2.json')
            #np.savetxt('best_params_xgb_classification.txt', np.array(good_features_xgb, dtype='str'), delimiter=',',fmt="%s")
        
        elif train_best_model:
            good_features_xgb = list(np.loadtxt('Classification_SimonGuldager_XGBoost_VariableList.txt', dtype = "str"))
            xgb_params = dict(objective = 'binary:logistic', min_split_loss = 0, min_child_weight = 1, subsample = 1,\
                                    colsample_bytree = 1, colsample_bylevel = 1, reg_lambda = 1, reg_alpha = 0, 
                                    eval_metric='logloss') #, early_stopping_rounds=120)
            #best_trial_params = dict(learning_rate=0.2131702912636896, n_estimators=86, max_depth=5)
            best_trial_params = {'n_estimators': 82, 'learning_rate': 0.1670222003534425, 'max_depth': 6}
            xgb_params.update(best_trial_params)
            print("Optimized parameters", xgb_params)
            ## Fit using the best parameters and evaluate results
            xgb_clf = xgb.XGBClassifier(**xgb_params)
            #xgb_clf.fit(X_train[good_features_xgb], y_train, eval_set = [(X_val[good_features_xgb], y_val)])

            cv_results = cross_validate(xgb_clf, X_train[good_features_xgb], y_train, scoring='neg_log_loss', cv=5) #, fit_params=xgb_params)
            print(-cv_results['test_score'], -cv_results['test_score'].mean())

            xgb_params = dict(objective = 'binary:logistic', min_split_loss = 0, min_child_weight = 1, subsample = 1,\
                                    colsample_bytree = 1, colsample_bylevel = 1, reg_lambda = 1, reg_alpha = 0, 
                                    eval_metric='logloss', early_stopping_rounds=120)
            best_trial_params = dict(learning_rate=0.2131702912636896, n_estimators=86, max_depth=5)
            xgb_params.update(best_trial_params)

            xgb_clf = xgb.XGBClassifier(**xgb_params)
            xgb_clf.fit(X_train[good_features_xgb], y_train, eval_set = [(X_val[good_features_xgb], y_val)])
            print("Best score:", xgb_clf.best_score)

            train_pred = xgb_clf.predict_proba(X_train[good_features_xgb])[:,1]
            val_pred = xgb_clf.predict_proba(X_val[good_features_xgb])[:,1]
    
            fpr_train, tpr_train, _ = roc_curve(y_train, train_pred)                  # False/True Positive Rate for our model
            fpr_val, tpr_val, _ = roc_curve(y_val, val_pred)  # False/True Positive Rate for Aleph NNbjet

            # We can now calculate the AUC scores of these ROC-curves:
            auc_score_train = auc(fpr_train,tpr_train)                       
            auc_score_val = auc(fpr_val, tpr_val)  

            print("Val acc. after param optimization", xgb_clf.score(X_val[good_features_xgb],y_val))
            print("Train acc. after param optimization ", xgb_clf.score(X_train[good_features_xgb],y_train))

            benchmark_stats = np.loadtxt('benchmark_performance_xgb.txt')
            print("Benchmark logloss/acc/AUC: ", benchmark_stats)


            #save best model
            xgb_clf.save_model('xgb_classifier4.json')
        else:
            good_features_xgb = list(np.loadtxt('Classification_SimonGuldager_XGBoost_VariableList.txt', dtype = "str"))
            xgb_clf = xgb.XGBClassifier()
            xgb_clf.load_model('xgb_classifier.json')
            print(xgb_clf.get_params)
            train_pred = xgb_clf.predict_proba(X_train[good_features_xgb])[:,1]
            val_pred = xgb_clf.predict_proba(X_val[good_features_xgb])[:,1]
    
            fpr_train, tpr_train, _ = roc_curve(y_train, train_pred)                  # False/True Positive Rate for our model
            fpr_val, tpr_val, _ = roc_curve(y_val, val_pred)  # False/True Positive Rate for Aleph NNbjet

            # We can now calculate the AUC scores of these ROC-curves:
            auc_score_train = auc(fpr_train,tpr_train)                       
            auc_score_val = auc(fpr_val, tpr_val)  

            log_loss_val = log_loss(y_val, val_pred)
            print("log_loss_val: ", log_loss_val)

            print("Val acc. after param optimization", xgb_clf.score(X_val[good_features_xgb],y_val))
            print("Train acc. after param optimization ", xgb_clf.score(X_train[good_features_xgb],y_train))

            test_pred = xgb_clf.predict_proba(test_features[good_features_xgb])[:,1]

            with open('Classification_SimonGuldager_XGBoost_Classification_SimonGuldager_XGBoost_VariableList.txt', 'w') as file:
                # Iterate over the array elements
                for i, value in enumerate(test_pred):
                    # Write the index and value separated by a comma
                    file.write(f'{i}, {value}\n')


            #np.savetxt('Classification_SimonGuldager_XGBoost_VarList1.txt')


        # Let's plot the ROC curves for these results:
        fig, ax = plt.subplots()

        ax.plot(fpr_train, tpr_train, label=f'XGB on train (AUC = {auc_score_train:5.3f})')
        ax.plot(fpr_val, tpr_val, label = f'XGB on val (AUC = {auc_score_val:5.3f})')
        ax.set(title='XGB ROC for train and val data', xlabel = 'FPR', ylabel = 'TPR')
        ax.legend()
        plt.show()

### MAIN ---------------------------------------------------------------------------------------

## TODO:
# Add normalization layer. Add batch norm. layers and perform inverse transform of data. ?? So maybe transform
# target var. before feeding it to network?
# GET TF up and running and est. necessary complexity [use hyperopt if necessary]
# discard features, either by feat importance or by using lightgbm's preffered.
# hyperopt
# produce results
# clustering

def main():
    ## Load data 
    train = load_data('data/train') #.iloc[:2000,:]
    test  = load_data('data/test') #.iloc[:2000,:]

    print (f'Shape of training data set: {train.shape}')
    print (f'Shape of test data set: {test.shape}')

    # store all feature names in list
    all_variables = ['actualInteractionsPerCrossing', 'averageInteractionsPerCrossing', 'correctedActualMu', 'correctedAverageMu', 'correctedScaledActualMu', 'correctedScaledAverageMu', 'NvtxReco', 'p_nTracks', 'p_pt_track', 'p_eta', 'p_phi', 'p_charge', 'p_qOverP', 'p_z0', 'p_d0', 'p_sigmad0', 'p_d0Sig', 'p_EptRatio', 'p_dPOverP', 'p_z0theta', 'p_etaCluster', 'p_phiCluster', 'p_eCluster', 'p_rawEtaCluster', 'p_rawPhiCluster', 'p_rawECluster', 'p_eClusterLr0', 'p_eClusterLr1', 'p_eClusterLr2', 'p_eClusterLr3', 'p_etaClusterLr1', 'p_etaClusterLr2', 'p_phiClusterLr2', 'p_eAccCluster', 'p_f0Cluster', 'p_etaCalo', 'p_phiCalo', 'p_eTileGap3Cluster', 'p_cellIndexCluster', 'p_phiModCalo', 'p_etaModCalo', 'p_dPhiTH3', 'p_R12', 'p_fTG3', 'p_weta2', 'p_Reta', 'p_Rphi', 'p_Eratio', 'p_f1', 'p_f3', 'p_Rhad', 'p_Rhad1', 'p_deltaEta1', 'p_deltaPhiRescaled2', 'p_TRTPID', 'p_TRTTrackOccupancy', 'p_numberOfInnermostPixelHits', 'p_numberOfPixelHits', 'p_numberOfSCTHits', 'p_numberOfTRTHits', 'p_numberOfTRTXenonHits', 'p_chi2', 'p_ndof', 'p_SharedMuonTrack', 'p_E7x7_Lr2', 'p_E7x7_Lr3', 'p_E_Lr0_HiG', 'p_E_Lr0_LowG', 'p_E_Lr0_MedG', 'p_E_Lr1_HiG', 'p_E_Lr1_LowG', 'p_E_Lr1_MedG', 'p_E_Lr2_HiG', 'p_E_Lr2_LowG', 'p_E_Lr2_MedG', 'p_E_Lr3_HiG', 'p_E_Lr3_LowG', 'p_E_Lr3_MedG', 'p_ambiguityType', 'p_asy1', 'p_author', 'p_barys1', 'p_core57cellsEnergyCorrection', 'p_deltaEta0', 'p_deltaEta2', 'p_deltaEta3', 'p_deltaPhi0', 'p_deltaPhi1', 'p_deltaPhi2', 'p_deltaPhi3', 'p_deltaPhiFromLastMeasurement', 'p_deltaPhiRescaled0', 'p_deltaPhiRescaled1', 'p_deltaPhiRescaled3', 'p_e1152', 'p_e132', 'p_e235', 'p_e255', 'p_e2ts1', 'p_ecore', 'p_emins1', 'p_etconeCorrBitset', 'p_ethad', 'p_ethad1', 'p_f1core', 'p_f3core', 'p_maxEcell_energy', 'p_maxEcell_gain', 'p_maxEcell_time', 'p_maxEcell_x', 'p_maxEcell_y', 'p_maxEcell_z', 'p_nCells_Lr0_HiG', 'p_nCells_Lr0_LowG', 'p_nCells_Lr0_MedG', 'p_nCells_Lr1_HiG', 'p_nCells_Lr1_LowG', 'p_nCells_Lr1_MedG', 'p_nCells_Lr2_HiG', 'p_nCells_Lr2_LowG', 'p_nCells_Lr2_MedG', 'p_nCells_Lr3_HiG', 'p_nCells_Lr3_LowG', 'p_nCells_Lr3_MedG', 'p_pos', 'p_pos7', 'p_poscs1', 'p_poscs2', 'p_ptconeCorrBitset', 'p_ptconecoreTrackPtrCorrection', 'p_r33over37allcalo', 'p_topoetconeCorrBitset', 'p_topoetconecoreConeEnergyCorrection', 'p_topoetconecoreConeSCEnergyCorrection', 'p_weta1', 'p_widths1', 'p_widths2', 'p_wtots1', 'p_e233', 'p_e237', 'p_e277', 'p_e2tsts1', 'p_ehad1', 'p_emaxs1', 'p_fracs1', 'p_DeltaE', 'p_E3x5_Lr0', 'p_E3x5_Lr1', 'p_E3x5_Lr2', 'p_E3x5_Lr3', 'p_E5x7_Lr0', 'p_E5x7_Lr1', 'p_E5x7_Lr2', 'p_E5x7_Lr3', 'p_E7x11_Lr0', 'p_E7x11_Lr1', 'p_E7x11_Lr2', 'p_E7x11_Lr3', 'p_E7x7_Lr0', 'p_E7x7_Lr1' ]

    print("No. of features: ", len(all_variables))

    # Split data into labels, targets and features
    train_features = train[all_variables]
    train_labels = train['Truth']
    train_target = train['p_truth_E']
    test_features = test[all_variables]

    print("Fraction of electrons in training data: ", (train_labels == 1).sum()/train_features.shape[0])

    # Decide whether to drop uninformative and highly correlated columns or use previous results
    drop_correlations = False
    corr_cutoff = 0.8
    if drop_correlations:
        feature_delete_list = []
        for feature in train_features.columns:
            if train_features[feature].min() == train_features[feature].max():
                feature_delete_list.append(feature)
                del train_features[feature]
        print("The following feature are identical for all entries and have been dropped: ")
        print(feature_delete_list)
        print("New shape of data feature matrix: ", train_features.shape)

       
        for i,col in enumerate(train_features.columns):
            if i % 10 == 0:
                print("Calculating for column ", i)
            corr = train_features.corr()
            if col not in train_features.columns:
                continue
            else:
                np.fill_diagonal(corr.values,0)
                vals = np.argwhere(np.abs(corr[col]) > corr_cutoff).flatten()
                if len(vals) != 0:
                    columns_to_drop = list(train_features.columns[vals])
                    feature_delete_list.extend(columns_to_drop)
                    train_features.drop(train_features.columns[vals],axis=1,inplace=True)

        print(f"If two features are correlated stronger than {corr_cutoff}, one of them is dropped")
        print("New shape of data feature matrix: ", train_features.shape)
        np.savetxt('bad_features.txt', np.array(feature_delete_list, dtype='str'), delimiter=',',fmt="%s")
    else:
        feature_delete_list = list(np.loadtxt('bad_features.txt', dtype = "str"))
        train_features.drop(feature_delete_list,inplace=True,axis=1)
    ## drop bad features from test data
    test_features.drop(feature_delete_list, inplace=True, axis=1)

    print (f'Shape of training data set after droppping uninformative features: {train_features.shape}')
    print (f'Shape of test data set after dropping uninformative features: {test_features.shape}')
    from sklearn.preprocessing import StandardScaler

    # Standardize input data
    if 1:
        scaler = StandardScaler()
        scaled_train_features_arr = scaler.fit_transform(train_features.values)
        scaled_test_features_arr = scaler.fit_transform(test_features.values)
        scaled_train_features = pd.DataFrame(scaled_train_features_arr, index = train_features.index, \
                                             columns = train_features.columns)
        scaled_test_features = pd.DataFrame(scaled_test_features_arr, index = test_features.index, \
                                             columns = test_features.columns)

    ## Split training data into training and validation data for classification
    X_train, X_val, y_train, y_val = train_test_split(scaled_train_features, train_labels, test_size=0.15, random_state=42)

    ## Split training data (only eletrons) into training and validation data for regression
    X_reg_train, X_reg_val, y_reg_train, y_reg_val = train_test_split(scaled_train_features[train_labels == 1], \
                                        train_target[train_labels == 1], test_size=0.15, random_state=42)

    print("No. of training and val. points: ", X_train.shape[0], X_val.shape[0])

    ################ CLASSIFICATION ##################
    ## APP 1: XGBoost
    xgb_classification, lgb_classification, mlp_classification = False, False, False
    if xgb_classification:
        xgb_classifier(X_train, X_val, y_train, y_val, scaled_test_features)
    elif lgb_classification:

        ## APP 2: LGBM 
        feature_importance, hyperoptimization, evaluate_best_model = False, False, False
        ## After some experimenting, it turns out that choosing the 15 best features as estimated by the rf model 
        # considerably worsens performance, whereas choosing xgb preffered features doesn't hinder performance. 
        # These have therefore been chosen.
        lgb_kwargs = dict(boosting_type='gbdt', num_leaves=120, \
                            max_depth=6, learning_rate=0.2, n_estimators=250, \
                            objective='binary', min_split_gain=0.0,\
                            min_child_samples=1, subsample = 1.0,
                            reg_alpha=0.0, reg_lambda=0.0, \
                            n_jobs=-1, importance_type = 'split') 
 
        # XGB's choice of features are better than that of LGBM
        good_features_lgb = list(np.loadtxt('Classification_SimonGuldager_XGBoost_VariableList.txt', dtype = "str"))
        np.savetxt('Classification_SimonGuldager_LGB_GBDT_VariableList.txt',\
                        np.array(good_features_lgb, dtype='str'), delimiter=',',fmt="%s")
        
        if 0:
            lgb_clf = lgb.LGBMClassifier(**lgb_kwargs)
            lgb_clf.fit(X_train[good_features_lgb], y_train, eval_set = [(X_val[good_features_lgb], y_val)], verbose=0)

            evaluate_classification_results(lgb_clf, X_train[good_features_lgb], \
                                            X_val[good_features_lgb], y_train, y_val, method_name='LGB RF', plot=True)


        if feature_importance:

            lgb_clf = lgb.LGBMClassifier(**lgb_kwargs)
            lgb_clf.fit(X_train, y_train, eval_set = [(X_val, y_val)], \
                        callbacks=[lgb.early_stopping(50)])
            
            print(lgb_clf.feature_importances_)
            
            evaluate_classification_results(lgb_clf, X_train, \
                                    X_val, y_train, y_val, method_name ='LGBM GDBT', plot=True)

            ## Step 2: Extract the 20 most important features
            good_features_lgb_indices = np.argsort(-lgb_clf.feature_importances_)[:15]
            good_features_lgb = X_train.columns[good_features_lgb_indices]
        
            # Save good features for lgb
            np.savetxt('Classification_SimonGuldager_LGB_GBDT_VariableList.txt',\
                        np.array(good_features_lgb, dtype='str'), delimiter=',',fmt="%s")
           
            # Fit again to ensure that performance is not hurt
            lgb_clf = lgb.LGBMClassifier(**lgb_kwargs)
            lgb_clf.fit(X_train[good_features_lgb], y_train, eval_set = [(X_val[good_features_lgb], y_val)], \
                        callbacks=[lgb.early_stopping(50)])

            print(lgb_clf.feature_importances_)

            benchmark_train, benchmark_val = evaluate_classification_results(lgb_clf, X_train[good_features_lgb], \
                                    X_val[good_features_lgb], y_train, y_val, method_name ='LGBM GDBT', plot=True)
            
            np.savetxt('benchmark_performance_lgb_clf.txt', benchmark_val)

        benchmark_stats_lgb = np.loadtxt('benchmark_performance_lgb_clf.txt')

        ## STEP 2: Hyperopt and avoid overtraining
        if hyperoptimization:

            # Load the best features
            good_features_lgb = list(np.loadtxt('Classification_SimonGuldager_XGBoost_VariableList.txt', dtype = "str"))

            # define wrapper holding the model parameters
            def lgb_clf_wrapper(trial):

                importance_types = ["gain", "split"]
                importance_type = trial.suggest_categorical("importance_type", importance_types)

                # set constant parameters
                lgb_clf_kwargs = dict(boosting_type='gbdt', min_split_gain=0.0,\
                                      reg_alpha=0.0, reg_lambda=0.0,
                                      objective='binary', n_jobs=-1)
                # set parameters to vary                  
                lgb_clf_kwargs_var =  {#'subsample': trial.suggest_float('subsample', 0.6,1),
                            'n_estimators': trial.suggest_int('n_estimators',250,400),
                            'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.4),
                            'max_depth': trial.suggest_int('max_depth', 4,10), \
                            'num_leaves': trial.suggest_int('num_leaves',50, 200), \
                            'subsample': trial.suggest_float('subsample', 0.75,1),\
                            'importance_type': importance_type,}
                
                lgb_clf_kwargs.update(lgb_clf_kwargs_var)
                return lgb_clf_kwargs

            # create optimization object
            study = optuna.create_study(direction="maximize",sampler=TPESampler(),pruner=MedianPruner(n_warmup_steps=50))
  
            # create objective function to pass and optimize
            restricted_objective = lambda trial: objective(trial, lgb.LGBMClassifier, lgb_clf_wrapper, \
                                scoring='neg_log_loss', X_train = X_train[good_features_lgb], y_train = y_train)
            study.optimize(restricted_objective, n_trials=30, show_progress_bar=False)
        
            print(study.best_trial.params)
            print(study.best_trial.values)

            lgb_clf_params = dict(boosting_type='gbdt', min_split_gain=0.0,\
                                      reg_alpha=0.0, reg_lambda=0.0,
                                      objective='binary', n_jobs=-1)
            
            lgb_clf_params.update(study.best_trial.params)
            print("Optimized parameters", lgb_clf_params)
            ## Fit using the best parameters and evaluate results
            lgb_clf = lgb.LGBMClassifier(**lgb_clf_params)
            lgb_clf.fit(X_train[good_features_lgb], y_train, eval_set = [(X_val[good_features_lgb], y_val)], verbose=0)

            evaluate_classification_results(lgb_clf, X_train[good_features_lgb], \
                                            X_val[good_features_lgb], y_train, y_val, method_name='LGB RF', plot=True)

            print("Benchmark stats for logloss, acc, AUC: ", benchmark_stats_lgb)

            # Save best parameters
            with open('best_params_lgb_clf.pkl', 'wb') as fp:
                pickle.dump(lgb_clf_params, fp)

            ## save beset model
            lgb_clf.booster_.save_model('lgb_classifier.json')


        if evaluate_best_model:
            # Load optimized parameters
            with open('best_params_lgb_clf.pkl', 'rb') as fp:
                lgb_clf_params = pickle.load(fp)
            print(lgb_clf_params)
            ## Fit using the best parameters and evaluate results
            lgb_clf = lgb.LGBMClassifier(**lgb_clf_params)
            lgb_clf.fit(X_train[good_features_lgb], y_train, eval_set = [(X_val[good_features_lgb], y_val)], verbose=0)

            evaluate_classification_results(lgb_clf, X_train[good_features_lgb], \
                                            X_val[good_features_lgb], y_train, y_val, method_name='LGB RF', plot=True)

            print("Benchmark stats for logloss, acc, AUC: ", benchmark_stats_lgb)

            # make predictions on test data
            test_pred = lgb_clf.predict_proba(scaled_test_features[good_features_lgb])[:,1]
     
            # save results to file
            with open('Classification_SimonGuldager_LGB_GBDT.txt', 'w') as file:
                # Iterate over the array elements
                for i, value in enumerate(test_pred):
                    # Write the index and value separated by a comma
                    file.write(f'{i}, {value}\n')
 
    if mlp_classification:
        explore_solutions, hyperoptimization, predict_test_labels = False, False, True

        good_features_mlp = list(np.loadtxt('Classification_SimonGuldager_XGBoost_VariableList.txt', dtype = "str"))
        Ninput = len(good_features_mlp)

        if explore_solutions:
            mlp_kwargs = dict(hidden_layer_sizes=(Ninput,64,128,256,256,128,64), activation='relu', solver='adam', alpha=0.0001,\
                            batch_size=128, learning_rate='constant', learning_rate_init=0.003, power_t=0.5, max_iter=30, \
                                shuffle=True, tol=0.0001, verbose=True, warm_start=False, momentum=0.9, \
                                    early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,\
                                        epsilon=1e-08, n_iter_no_change=10)

            mlp = MLPClassifier(**mlp_kwargs)
            mlp.fit(X_train[good_features_mlp], y_train)

            evaluate_classification_results(mlp, X_train[good_features_mlp], X_val[good_features_mlp], y_train, y_val, method_name='MLP', plot=True)
        if hyperoptimization:
            from scipy.stats import randint, poisson,uniform
            mlp_kwargs = dict(hidden_layer_sizes=(Ninput, 64, 128, 256, 256, 128, 64), activation='relu', solver='adam', alpha=0.0001,\
                            batch_size=128, learning_rate='constant', learning_rate_init = 0.003, max_iter=20, \
                                shuffle=True, tol=0.0001, verbose=True, warm_start=False, momentum=0.9, \
                                    early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,\
                                        epsilon=1e-08, n_iter_no_change=10)
            mlp_kwargs_var = {
                'learning_rate_init': uniform(0.0007,0.01),
                'batch_size': [64, 128]
            }

            mlp_kwargs.update(mlp_kwargs_var)
            mlp = MLPClassifier(**mlp_kwargs)

            mlp = RandomizedSearchCV(mlp, mlp_kwargs_var, n_iter = 20, scoring = 'neg_log_loss', \
                                            cv = 4, n_jobs = -1).fit(X_train[good_features_mlp], y_train)

            print("Best Param for GS", mlp.best_params_)
            print("CV score for GS", mlp.best_score_)

            evaluate_classification_results(mlp, X_train[good_features_mlp], X_val[good_features_mlp], y_train, y_val, method_name='MLP', plot=True)
        if predict_test_labels:
            ## update kwargs with best params
            mlp_kwargs = dict(hidden_layer_sizes=(Ninput, 64, 128, 256, 256, 128, 64), activation='relu', solver='adam', alpha=0.0001,\
                            batch_size=128, learning_rate='constant', learning_rate_init = 0.003, max_iter=25, \
                                shuffle=True, tol=0.0001, verbose=True, warm_start=False, momentum=0.9, \
                                    early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,\
                                        epsilon=1e-08, n_iter_no_change=5)
            mlp_kwargs['batch_size'] = 1024
            mlp_kwargs['learning_rate_init'] = 0.00173

            mlp = MLPClassifier(**mlp_kwargs)
            mlp.fit(X_train[good_features_mlp], y_train)

            evaluate_classification_results(mlp, X_train[good_features_mlp], X_val[good_features_mlp], y_train, y_val, method_name='MLP', plot=True)

            # Predict test labels
            test_pred = mlp.predict_proba(scaled_test_features[good_features_mlp])[:,1]

            from joblib import dump, load
            dump(mlp, 'mlp.joblib')

            # save results to file
            if 0:
                with open('Classification_SimonGuldager_SKLEARN_MLP.txt', 'w') as file:
                    # Iterate over the array elements
                    for i, value in enumerate(test_pred):
                        # Write the index and value separated by a comma
                        file.write(f'{i}, {value}\n')



    ################## REGRESSION ####################

    if 0:
        # Do a minimal implementation of XGBoost

        xgb_params = dict(objective='reg:squarederror', learning_rate=0.3, min_split_loss = 0, min_child_weight = 1, subsample = 1,\
                                colsample_bytree = 1, colsample_bylevel = 1,
                                            reg_lambda = 1, reg_alpha = 0, max_depth=14, eval_metric='mape', n_estimators=450,\
                                                early_stopping_rounds=40)

        xgb_clf = xgb.XGBRegressor(**xgb_params)
        xgb_clf.fit(X_reg_train, y_reg_train, eval_set = [(X_reg_val, y_reg_val)])
        
        train_pred = xgb_clf.predict(X_reg_train)
        val_pred = xgb_clf.predict(X_reg_val)

        evaluate_predictions(val_pred, y_reg_val, label='val')
        evaluate_predictions(train_pred, y_reg_train, label='train')

        ## Step 2: Extract the 15 most important features
        #good_features_xgb_indices = np.argwhere(xgb_clf.feature_importances_ > 0.01).flatten()
        good_features_xgb_indices = np.argsort(-xgb_clf.feature_importances_)[:20]
        good_features_xgb = X_train.columns[good_features_xgb_indices]
        
        # Save good features for xgb
        # np.savetxt('Classification_SimonGuldager_XGBoost_VariableList.txt', np.array(good_features_xgb, dtype='str'), delimiter=',',fmt="%s")

        # Fit again to ensure that performance is not hurt
        xgb_clf = xgb.XGBRegressor(**xgb_params)
        xgb_clf.fit(X_reg_train[good_features_xgb], y_reg_train,  eval_set = [(X_reg_val[good_features_xgb], y_reg_val)])
        
        train_pred = xgb_clf.predict(X_reg_train[good_features_xgb])
        val_pred = xgb_clf.predict(X_reg_val[good_features_xgb])

        evaluate_predictions(val_pred, y_reg_val, label='val')
        evaluate_predictions(train_pred, y_reg_train, label='train')

    lgb_regression, tf_regression, rf_regression = False, False, False
    if lgb_regression:
 
        ## APP 2: LGBM Boosted trees
        feature_importance, hyperoptimization, evaluate_best_model = False, False, False
        
        lgb_kwargs = dict(boosting_type='gbdt', objective='regression', \
                          n_jobs=-1, reg_alpha=0.1, reg_lambda=0.0,
                             min_split_gain=0.0, min_child_samples=5, \
                            max_depth=6, learning_rate=0.1, n_estimators=250,
                             subsample = 1, num_leaves=50, importance_type='split')  #vs 'gain'
     
        #good_features_lgb = list(np.loadtxt('Classification_SimonGuldager_XGBoost_VariableList.txt', dtype = "str"))

        if feature_importance:

            lgb_clf = lgb.LGBMRegressor(**lgb_kwargs)
            lgb_clf.fit(X_reg_train, y_reg_train, eval_set = [(X_reg_val, y_reg_val)], \
                        eval_metric='mae', callbacks=[lgb.early_stopping(50)])
            
            print(lgb_clf.feature_importances_)
            
            ##EVAL PÅ MAE, R2, mean squared log error
            ## Plot noget pred vs true, alternativt fordelingerne oveni hinanden eller rel error per punkt.
            ## mod eval function til reg
            evaluate_predictions(lgb_clf.predict(X_reg_val), y_reg_val, label = 'Val')

            ## Step 2: Extract the 20 most important features
            good_features_lgb_indices = np.argsort(-lgb_clf.feature_importances_)[:20]
            good_features_lgb = X_reg_train.columns[good_features_lgb_indices]
        
            # Save good features for lgb
            np.savetxt('Regression_SimonGuldager_LGB_GBDT_VariableList.txt', np.array(good_features_lgb, dtype='str'), delimiter=',',fmt="%s")
            # Load the best features
            good_features_lgb = list(np.loadtxt('Regression_SimonGuldager_LGB_GBDT_VariableList.txt', dtype = "str"))
            # Fit again to ensure that performance is not hurt
            lgb_clf = lgb.LGBMRegressor(**lgb_kwargs)
            lgb_clf.fit(X_reg_train[good_features_lgb], y_reg_train, eval_set = [(X_reg_val[good_features_lgb], y_reg_val)], \
                        eval_metric='l2', callbacks=[lgb.early_stopping(50)])

            print(lgb_clf.feature_importances_)

            # Evaluate predictions
            y_pred_val = lgb_clf.predict(X_reg_val[good_features_lgb])
            y_pred_train = lgb_clf.predict(X_reg_train[good_features_lgb])
            benchmark_stats_lgb = evaluate_predictions(y_pred_val, y_reg_val, label='val') 
            benchmark_stats_lgb_train = evaluate_predictions(y_pred_train, y_reg_train, label='train') 
            np.savetxt('benchmark_performance_lgb.txt', benchmark_stats_lgb)
            
  

            visualize_regression(y_pred_train, y_pred_val, y_reg_train, y_reg_val, model_name = 'LGB GDBT')

        benchmark_stats_lgb = np.loadtxt('benchmark_performance_lgb.txt')

        ## STEP 2: Hyperopt and avoid overtraining
        if hyperoptimization:

            # Load the best features
            good_features_lgb = list(np.loadtxt('Regression_SimonGuldager_LGB_GBDT_VariableList.txt', dtype = "str"))
           
            print(len(good_features_lgb))
            # define wrapper holding the model parameters
            def lgb_wrapper(trial):
                # set constant parameters
                lgb_kwargs = dict(boosting_type='gbdt', min_split_gain=0.0,\
                                      reg_alpha=0.0, reg_lambda=0.0,
                                      objective='regression', n_jobs=-1)
                # set parameters to vary                  

                importance_types = ["gain", "split"]
                importance_type = trial.suggest_categorical("importance_type", importance_types)

                if 1:
                    lgb_kwargs_var =  {#'subsample': trial.suggest_float('subsample', 0.6,1),
                                'n_estimators': trial.suggest_int('n_estimators',150,500),
                                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.25, log=True),
                                'max_depth': trial.suggest_int('max_depth', 4,12), \
                                'num_leaves': trial.suggest_int('num_leaves',30,100), \
                                'min_child_samples': trial.suggest_int('min_child_samples', 1,20),\
                                'subsample': trial.suggest_float('subsample', 0.8, 1),\
                                'importance_type': importance_type}
                if 0:
                    lgb_kwargs_var =  {#'subsample': trial.suggest_float('subsample', 0.6,1),
                                    'n_estimators': trial.suggest_int('n_estimators',200,1000),
                                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                                    'max_depth': trial.suggest_int('max_depth', 4,15), \
                                    #'num_leaves': trial.suggest_int('num_leaves',30,200), \
                                # 'min_child_samples': trial.suggest_int('min_child_samples', 1,20),\
                                    'subsample': trial.suggest_float('subsample', 0.8, 1),\
                                    }#'importance_type': importance_type}

                lgb_kwargs.update(lgb_kwargs_var)
                return lgb_kwargs

            # create optimization object
            study = optuna.create_study(direction="maximize",sampler=TPESampler(),pruner=MedianPruner(n_warmup_steps=50))
  
            # create objective function to pass and optimize
            restricted_objective = lambda trial: objective(trial, lgb.LGBMRegressor, lgb_wrapper, \
                                scoring='neg_mean_absolute_percentage_error', \
                                X_train = X_reg_train[good_features_lgb], y_train = y_reg_train)
            study.optimize(restricted_objective, n_trials=50, show_progress_bar=False)
        
            print(study.best_trial.params)
            print(study.best_trial.values)

            # Train model with optimized parameters
            lgb_params = dict(boosting_type='gbdt', min_split_gain=0.0,\
                                      reg_alpha=0.0, reg_lambda=0.0,
                                      objective='regression', n_jobs=-1)
            
            lgb_params.update(study.best_trial.params)
            print("Optimized parameters", lgb_params)
            ## Fit using the best parameters and evaluate results
            lgb_clf = lgb.LGBMRegressor(**lgb_params)
            lgb_clf.fit(X_reg_train[good_features_lgb], y_reg_train, eval_set = [(X_reg_val[good_features_lgb], y_reg_val)], verbose=0)

            # Evaluate and visualize predictions
            y_pred_val = lgb_clf.predict(X_reg_val[good_features_lgb])
            y_pred_train = lgb_clf.predict(X_reg_train[good_features_lgb])
            benchmark_stats_lgb = evaluate_predictions(y_pred_val, y_reg_val, label='val') 
            benchmark_stats_lgb = evaluate_predictions(y_pred_train, y_reg_train, label='train') 
            

            visualize_regression(y_pred_train, y_pred_val, y_reg_train, y_reg_val, model_name = 'LGB GDBT')

            # Save best parameters
            with open('best_params_lgb.pkl', 'wb') as fp:
                pickle.dump(dict, fp)
                print('dictionary saved successfully to file')

            ## save best model
            lgb_clf.booster_.save_model('lgb_regressor.json')

        if evaluate_best_model:
            # Load best parameters
            with open('best_params_lgb.pkl', 'rb') as fp:
                lgb_best_params = pickle.load(fp)
     
            # Load the best features
            good_features_lgb = list(np.loadtxt('Regression_SimonGuldager_LGB_GBDT_VariableList.txt', dtype = "str"))
           
            # Load best model
            lgb_clf = lgb.Booster(model_file='lgb_regressor.json')

            # make predictions on test data
            test_pred = lgb_clf.predict(scaled_test_features[good_features_lgb])
            if 0:
                # save results to file
                with open('Regression_SimonGuldager_LGB_GDBT.txt', 'w') as file:
                    # Iterate over the array elements
                    for i, value in enumerate(test_pred):
                        # Write the index and value separated by a comma
                        file.write(f'{i}, {value}\n')
 
    if tf_regression:
        # Build neural network model
        good_features_tf = list(np.loadtxt('Regression_SimonGuldager_LGB_GBDT_VariableList.txt', dtype = "str"))
        Ncolumns = len(X_reg_train[good_features_tf].columns)

        def baseline_model(input_length = Ncolumns, kernel_initializer = 'random_normal', learning_rate = 0.001,\
                           batch_norm_momentum = 0.99,  verbose = 1):
            l1 = False
            kwargs = {'kernel_initializer': kernel_initializer, 'activation': 'relu'}
            if l1:
                kwargs.update({'kernel_regularizer':regularizers.L1(0.0001)}), \
              #   'bias_regularizer':regularizers.L1(1e-5),
               #  'activity_regularizer':regularizers.L1(1e-5)})

            model = models.Sequential()
           
            model.add(layers.Dense(input_length, input_shape = (input_length,), **kwargs, ))
            model.add(layers.Dense(64,**kwargs))
            tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, center=True, scale=True)
            model.add(layers.Dense(128,**kwargs))
            tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, center=True, scale=True)
            model.add(layers.Dense(256, **kwargs))
            tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, center=True, scale=True)
            model.add(layers.Dense(128,**kwargs))
            tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum, center=True, scale=True)
            model.add(layers.Dense(64,**kwargs))

            model.add(layers.Dense(1, activation = 'linear')) #, activation = 'sigmoid'))
            model.build()

            if verbose:
                print(model.summary())

            # Set learning schedule parameters
            boundaries = [10_000, 20_000, 30_000]
            values = [1.3 * learning_rate, learning_rate, 0.7 * learning_rate, 0.4 * learning_rate]
        
            # Compile
            metric =tf.keras.metrics.MeanAbsolutePercentageError() #[tf.keras.metrics.MeanSquaredError()] [tf.keras.metrics.LogCoshError()]
            loss = tf.keras.losses.MeanAbsolutePercentageError() # 'mse' #tf.keras.losses.LogCosh() # tf.keras.losses.MeanSquaredLogarithmicError()
            kwargs_ann = {'optimizer': tf.keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)), \
                        'loss': loss, 'metrics': metric}   # 'loss': tf.keras.losses.BinaryCrossentropy(from_logits=True),
            model.compile(**kwargs_ann) 
            return model
    
        def model_wrapper():
            model = baseline_model(input_length = Ncolumns, learning_rate=learning_rate)
            return model


        batch_size = 128 #1024
        Nepochs = 70
        learning_rate = 0.008

        y_reg_train_norm = y_reg_train
        y_reg_val_norm = y_reg_val

        Ncolumns = len(X_reg_train[good_features_tf].columns)
        model = baseline_model(input_length = Ncolumns, learning_rate=learning_rate)

        #base_model = baseline_model(input_length = Ncolumns, \
         #                           kernel_initializer='random_normal', learning_rate = 0.001)
       # history = model.fit(X_reg_train, y_reg_train, epochs=Nepochs, batch_size = batch_size, \
        #                    validation_data=(X_reg_val, y_reg_val))

        tf_reg = KerasRegressor(build_fn=model_wrapper, epochs=Nepochs, batch_size=batch_size ,verbose=1)  
        if 0:
            cv_results = cross_validate(tf_reg, X_reg_train[good_features_tf], y_reg_train, scoring='neg_mean_absolute_percentage_error', cv=5) #, fit_params=xgb_params)
            print(cv_results['test_score'])
            #return cv_results['test_score'].mean()

        history = tf_reg.fit(X_reg_train[good_features_tf], y_reg_train, validation_data=(X_reg_val[good_features_tf], y_reg_val))
        training_loss = history.history['loss']
        validation_loss = history.history['val_loss']

        fig, ax =  plt.subplots()
        ax.plot(training_loss,label = 'training loss')
        ax.plot(training_loss,'o')
        ax.plot(validation_loss, label = 'validation loss')
        ax.plot(validation_loss, 'o')
        ax.legend()
    
        # Make predictions on test data
        test_pred = tf_reg.predict(scaled_test_features[good_features_tf])


        # save results to file
        with open('Regression_SimonGuldager_Tensorflow_NN.txt', 'w') as file:
            # Iterate over the array elements
            for i, value in enumerate(test_pred):
                # Write the index and value separated by a comma
                file.write(f'{i}, {value}\n')

    if rf_regression:
        good_features_rf = list(np.loadtxt('Regression_SimonGuldager_LGB_GBDT_VariableList.txt', dtype = "str"))

        ## These params have already been optimized (in terms of n_estimators, max_depth, max_samples)
        rf_kwargs = dict(n_estimators=356, criterion='squared_error', max_depth=22, \
                 min_samples_split=2, min_samples_leaf=15,  \
                 max_features='sqrt', max_leaf_nodes=None, \
                 bootstrap=True, n_jobs=-1, verbose=0,\
                  ccp_alpha=0.0, max_samples=0.826)
      
        rf = RandomForestRegressor(**rf_kwargs)
        rf.fit(X_reg_train[good_features_rf], y_reg_train)

        train_pred = rf.predict(X_reg_train[good_features_rf])
        val_pred = rf.predict(X_reg_val[good_features_rf])

        evaluate_predictions(train_pred, y_reg_train, label = 'train', quantiles = None, verbose = 1)
        evaluate_predictions(val_pred, y_reg_val, label = 'val', quantiles = None, verbose = 1)

   
        # Make test data predictions
        # make predictions on test data
        test_pred = rf.predict(scaled_test_features[good_features_rf])
        
        # Save model
        from joblib import dump
        dump(rf, 'rf.joblib')
        
        if 0:
            # save results to file
            with open('Regression_SimonGuldager_SKLearn_RandomForest.txt', 'w') as file:
                # Iterate over the array elements
                for i, value in enumerate(test_pred):
                    # Write the index and value separated by a comma
                    file.write(f'{i}, {value}\n')
 


    ################ CLUSTERING #######################
    kmeans, birch, bisection = False, False, False

    good_features_clustering = list(np.loadtxt('Clustering_SimonGuldager_SKLearn_MiniBatchKMeans_VariableList.txt', dtype = "str"))
    min_clusters, max_clusters = 3, 50

    if kmeans:

        find_best_features, find_best_n_cluster, predict_test_labels = False, False, True
        
        if find_best_features:

            ### STEP 1: Use 5 best params. Try to estimate from lgm classifier

            xgb_params = dict(objective='binary:logistic', learning_rate=0.3, min_split_loss = 0, min_child_weight = 1, subsample = 1,\
                                    colsample_bytree = 1, colsample_bylevel = 1,
                                                reg_lambda = 1, reg_alpha = 0, max_depth=4, eval_metric='logloss', n_estimators=50,)
            
            good_features_xgb = list(np.loadtxt('Classification_SimonGuldager_XGBoost_VariableList.txt', dtype = "str"))

            # Step 1: Fit on all (non-correlated) features
            xgb_clf = xgb.XGBClassifier(**xgb_params)
            xgb_clf.fit(X_train, y_train)
            evaluate_classification_results(xgb_clf, X_train, X_val, y_train, y_val, method_name='XGB')
            
            # Extract the best 5 features
            good_features_xgb_indices = np.argsort(- xgb_clf.feature_importances_)[:5]
            good_features_xgb = X_train.columns[good_features_xgb_indices]

            # Fit again to evaluate performance loss
            xgb_clf = xgb.XGBClassifier(**xgb_params)
            xgb_clf.fit(X_train[good_features_xgb], y_train)
            print(xgb_clf.n_features_in_)
            print(good_features_xgb)

            evaluate_classification_results(xgb_clf, X_train[good_features_xgb], X_val[good_features_xgb], y_train, y_val, method_name='XGB')

            # Save best features
            np.savetxt('Clustering_SimonGuldager_KMiniBatch_VariableList.txt', \
                    np.array(good_features_xgb, dtype='str'), delimiter=',',fmt="%s")
        else: 
            good_features_clustering = list(np.loadtxt('Clustering_SimonGuldager_SKLearn_MiniBatchKMeans_VariableList.txt', dtype = "str"))


        ## KMEANS and KMINIBATCH
        params_kmb = dict(n_clusters=8, init='k-means++', max_iter=100, batch_size=1024, verbose=0,\
                    compute_labels=True, random_state=42, tol=0.0, max_no_improvement=25, \
                        init_size=None, n_init=5, reassignment_ratio=0.01)

        params_km = dict(n_clusters=8, init='k-means++', n_init=5, max_iter=300, \
                        tol=0.0001, verbose=0, random_state=42, copy_x=True, algorithm='lloyd')

        
        inertia_arr = np.zeros([2,max_clusters - min_clusters])

        if find_best_n_cluster:
            
            acc_list = np.empty_like(inertia_arr)
            val_acc_list = np.empty_like(acc_list)
            sil_score_list = np.empty_like(val_acc_list)

            for n in np.arange(min_clusters, max_clusters):
                    print(n)
                    params_kmb['n_clusters'] = n
                    kmb = MiniBatchKMeans(**params_kmb)
                    kmb.fit(X_train[good_features_clustering])
                    inertia_arr[0, n - min_clusters] = kmb.inertia_
                    kmb_labels = kmb.labels_
                    kmb_acc, kmb_acc_arr, kmb_w_arr = clustering_accuracy(X_train[good_features_clustering], y_train, kmb_labels)
                    kmb_val_labels = kmb.predict(X_val[good_features_clustering])
                    kmb_val_acc,_,_ = clustering_accuracy(X_val[good_features_clustering], y_val, kmb_val_labels)
                    

                    val_acc_list[0, n - min_clusters] = kmb_val_acc
                    acc_list[0, n - min_clusters] = kmb_acc
                    print("kmb\n")
                    print(kmb_acc_arr)
                    print(kmb_w_arr)
                    params_km['n_clusters'] = n
                    km = KMeans(**params_km)
                    km.fit(X_train[good_features_clustering])
                    inertia_arr[1, n - min_clusters] = km.inertia_
                    km_labels = km.labels_
                    km_acc, km_acc_arr, km_w_arr = clustering_accuracy(X_train[good_features_clustering], y_train, km_labels)
                    acc_list[1, n - min_clusters] = km_acc

                    km_val_labels = km.predict(X_val[good_features_clustering])
                    km_val_acc,_,_ = clustering_accuracy(X_val[good_features_clustering], y_val, km_val_labels)

                    val_acc_list[1, n - min_clusters] = km_val_acc

                    sil_score_list[0, n - min_clusters] = silhouette_score(X_train[good_features_clustering].values, \
                                                                        kmb_labels, sample_size = 20_000)
                    sil_score_list[1, n - min_clusters] = silhouette_score(X_val[good_features_clustering].values, kmb_val_labels,)

                    print("km\n")
                    print(km_acc_arr)
                    print(km_w_arr)
                    

                    print("\nkmb/km acc: ", kmb_acc, km_acc)
                    print("kmb/km acc val: ", kmb_val_acc, km_val_acc)

                    
            fig0, ax0 = plt.subplots()
            ax00 = ax0.twinx()
            ax0.plot(np.arange(min_clusters, max_clusters), inertia_arr[0,:], 'x', label = 'Minibatch KMeans')
            ax0.plot(np.arange(min_clusters, max_clusters), inertia_arr[1,:], 'o', label = 'Minibatch KMeans')
            ax0.set(xlabel = 'Iteration', ylabel = 'Inertia')

            ax00.plot(np.arange(min_clusters, max_clusters), sil_score_list[0,:], '--g', label = 'Mimnibatch Train Sil', lw=1)
            ax00.plot(np.arange(min_clusters, max_clusters), sil_score_list[1,:], '-r', label = 'KMeans val Sil',)

            ax00.plot(np.arange(min_clusters, max_clusters), acc_list[0,:], '--b', label = 'Mimnibatch mean acc.', lw = 1,)
            ax00.plot(np.arange(min_clusters, max_clusters), acc_list[1,:], '--r', label = 'KMeans mean acc.', lw = 1,)
            ax00.plot(np.arange(min_clusters, max_clusters), val_acc_list[0,:], '-b', label = 'Mimnibatch mean acc. val', lw = 1.5,)
            ax00.plot(np.arange(min_clusters, max_clusters), val_acc_list[1,:], '-r', label = 'KMeans mean acc. val', lw = 1.5,)
            ax0.legend()
            ax00.legend()
            
            # Find best no of clusters
            kmb_best_n = np.argmax(acc_list[0,:])
            km_best_n = np.argmax(acc_list[1, :])

        if predict_test_labels:
            best_n_cluster_km = 16
            best_n_cluster_kmb = 18

            params_km['n_clusters'] = best_n_cluster_km
            params_kmb['n_clusters'] = best_n_cluster_kmb
        
            
            kmb = MiniBatchKMeans(**params_kmb)
            kmb.fit(X_train[good_features_clustering])


            kmb_labels = kmb.labels_
            kmb_acc, kmb_acc_arr, kmb_w_arr = clustering_accuracy(X_train[good_features_clustering], y_train, kmb_labels)
            kmb_val_labels = kmb.predict(X_val[good_features_clustering])
            kmb_val_acc,kmb_val_arr,kmb_val_w_arr = clustering_accuracy(X_val[good_features_clustering], y_val, kmb_val_labels)

            
            sil_score_train = silhouette_score(X_train[good_features_clustering].values, kmb_labels, sample_size = 20_000)
            sil_score_val = silhouette_score(X_val[good_features_clustering].values, kmb_val_labels,)
   
            print(sil_score_train, sil_score_val)

            if 1:
                print("kmb\n")
                print(kmb_acc_arr)
                print(kmb_w_arr)
                print("kmb val\n")
                print(kmb_val_arr)
                print(kmb_val_w_arr)
                print("acc/ val acc", kmb_acc, kmb_val_acc)


            ## Finally, predict test data clustering labels
            kmb_test_labels = kmb.predict(scaled_test_features[good_features_clustering])
            print(kmb.inertia_)

            #save model
            from joblib import dump, load
            dump(kmb, 'KMinibatch.joblib')

            if 0:
                # save results to file
                with open('Clustering_SimonGuldager_SKLearn_MiniBatchKMeans.txt', 'w') as file:
                    # Iterate over the array elements
                    for i, value in enumerate(kmb_test_labels):
                        # Write the index and value separated by a comma
                        file.write(f'{i}, {int(value)}\n')

    if birch:
        find_best_n_cluster, find_best_params, predict_test_clusters = False, False, True
        

        params_birch = dict(threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True)
      
        acc_list = np.empty([2, max_clusters - min_clusters])
        sil_score_list = np.empty_like(acc_list)


        if find_best_n_cluster:
            for n in np.arange(min_clusters, max_clusters):
                    print("Iteration ", n)
                    params_birch['n_clusters'] = n
                    birch = Birch(**params_birch)
                    birch.fit(X_train[good_features_clustering])
                    labels = birch.labels_
                    acc, acc_arr, w_arr = clustering_accuracy(X_train[good_features_clustering], y_train, labels)
                    val_labels = birch.predict(X_val[good_features_clustering])
                    val_acc,_,_ = clustering_accuracy(X_val[good_features_clustering], y_val, val_labels)
                    
                    acc_list[1, n - min_clusters] = val_acc
                    acc_list[0, n - min_clusters] = acc

                    sil_score_list[0, n - min_clusters] = silhouette_score(X_train[good_features_clustering].values, \
                                                                        labels, sample_size = 20_000)
                    sil_score_list[1, n - min_clusters] = silhouette_score(X_val[good_features_clustering].values, val_labels,)

                    
            fig0, ax0 = plt.subplots()
            ax00 = ax0.twinx()
            ax0.plot(np.arange(min_clusters, max_clusters), acc_list[0,:], 'gx', label = 'Train acc')
            ax0.plot(np.arange(min_clusters, max_clusters), acc_list[1,:], 'ro', label = 'Val acc')
            ax0.set(xlabel = 'Iteration', ylabel = 'Inertia', title = 'Birch')

            ax00.plot(np.arange(min_clusters, max_clusters), sil_score_list[0,:], '--g', label = 'Train Sil', lw=1)
            ax00.plot(np.arange(min_clusters, max_clusters), sil_score_list[1,:], '-r', label = 'Val Sil',)

            ax0.legend()
            ax00.legend()

        if find_best_params:
            best_n_cluster_birch = 43
            params_birch['n_clusters'] = best_n_cluster_birch
            # Coarse scan over branching factor and threshold

            threshold_list = np.arange(0.4,1.4,0.2)
            branching_factor = np.arange(50,300,50)
            Niterations = len(threshold_list) * len(branching_factor)

            acc_list = []
            val_acc_list = []
            sil_score_list = []
            val_sil_score_list = []
            it = 0
            for threshold in threshold_list:
                params_birch['threshold'] = threshold
                for branch in branching_factor:
                    it += 1
                    params_birch['branching_factor'] = branch
                    print(it)
                
                    birch = Birch(**params_birch)
                    birch.fit(X_train[good_features_clustering])
                    labels = birch.labels_
                    acc,_, _ = clustering_accuracy(X_train[good_features_clustering], y_train, labels)
                    val_labels = birch.predict(X_val[good_features_clustering])
                    val_acc,_,_ = clustering_accuracy(X_val[good_features_clustering], y_val, val_labels)
                    
                    acc_list.append(acc)
                    val_acc_list.append(val_acc)

                    sil_score_list.append(  silhouette_score(X_train[good_features_clustering].values, 
                                                                        labels, sample_size = 20_000))
                    val_sil_score_list.append( silhouette_score(X_val[good_features_clustering].values, val_labels,))

                    
            fig0, ax0 = plt.subplots()
            ax00 = ax0.twinx()
            ax0.plot(np.arange(Niterations), acc_list, 'gx', label = 'Train acc')
            ax0.plot(np.arange(Niterations), val_acc_list, 'ro', label = 'Val acc')
            ax0.set(xlabel = 'Iteration', ylabel = 'Inertia', title = 'Birch')

            ax00.plot(np.arange(Niterations), sil_score_list, '--g', label = 'Train Sil', lw=1)
            ax00.plot(np.arange(Niterations), val_sil_score_list, '-r', label = 'Val Sil',)

            ax0.legend()
            ax00.legend()

        if predict_test_clusters:
            best_n_cluster_birch = 43
            best_threshold = 0.4
            best_branch = 250
            params_birch['n_clusters'] = best_n_cluster_birch
            params_birch['threshold'] = best_threshold
            params_birch['branching_factor'] = best_branch

            # Fit fit best parameters
            birch = Birch(**params_birch)
            birch.fit(X_train[good_features_clustering])

            # Calc acc and silhouette score
            labels = birch.labels_
            acc,_, _ = clustering_accuracy(X_train[good_features_clustering], y_train, labels)
            val_labels = birch.predict(X_val[good_features_clustering])
            val_acc,_,_ = clustering_accuracy(X_val[good_features_clustering], y_val, val_labels)

            sil_score = silhouette_score(X_train[good_features_clustering].values, 
                                                                labels, sample_size = 20_000)
            val_sil_score = silhouette_score(X_val[good_features_clustering].values, val_labels,)

            print("train/val acc: ", acc, val_acc)
            print("train/val sil: ", sil_score, val_sil_score)

            # Predict clusters on test data 
            ## Finally, predict test data clustering labels
            birch_test_labels = birch.predict(scaled_test_features[good_features_clustering])
    
            #save model
            from joblib import dump, load
            dump(birch, 'birch.joblib')

            if 1:
                # save results to file
                with open('Clustering_SimonGuldager_SKLearn_Birch.txt', 'w') as file:
                    # Iterate over the array elements
                    for i, value in enumerate(birch_test_labels):
                        # Write the index and value separated by a comma
                        file.write(f'{i}, {int(value)}\n')

    if bisection:
        find_best_n_cluster, predict_test_labels = False, False

        min_clusters, max_clusters = 3, 50

        params_bisec = dict(n_clusters=8, init='k-means++', n_init=1, random_state=42\
                            , max_iter=300, verbose=0, tol=0.0001, copy_x=True, algorithm='lloyd', \
                                bisecting_strategy='largest_cluster')

        acc_list = np.empty([2, max_clusters - min_clusters])
        sil_score_list = np.empty_like(acc_list)
        inertia_list = []
  
        if find_best_n_cluster:
            for n in np.arange(min_clusters, max_clusters):
                    print("Iteration ", n)
                    params_bisec['n_clusters'] = n
                    bisec = BisectingKMeans(**params_bisec)
                    bisec.fit(X_train[good_features_clustering])
                    inertia_list.append(bisec.inertia_)

                    labels = bisec.labels_
                    acc, acc_arr, w_arr = clustering_accuracy(X_train[good_features_clustering], y_train, labels)
                    val_labels = bisec.predict(X_val[good_features_clustering])
                    val_acc,_,_ = clustering_accuracy(X_val[good_features_clustering], y_val, val_labels)
                    
                    acc_list[1, n - min_clusters] = val_acc
                    acc_list[0, n - min_clusters] = acc

                    sil_score_list[0, n - min_clusters] = silhouette_score(X_train[good_features_clustering].values, \
                                                                        labels, sample_size = 20_000)
                    sil_score_list[1, n - min_clusters] = silhouette_score(X_val[good_features_clustering].values, val_labels,)

                    
            fig0, ax0 = plt.subplots()
            ax00 = ax0.twinx()
            ax0.plot(np.arange(min_clusters, max_clusters), inertia_list, 'g-', label = 'Inertia')
            ax00.plot(np.arange(min_clusters, max_clusters), acc_list[0,:], 'gx', label = 'Train acc')
            ax00.plot(np.arange(min_clusters, max_clusters), acc_list[1,:], 'ro', label = 'Val acc')
            ax0.set(xlabel = 'Iteration', ylabel = 'Inertia', title = 'bisec')

            ax00.plot(np.arange(min_clusters, max_clusters), sil_score_list[0,:], '--g', label = 'Train Sil', lw=1)
            ax00.plot(np.arange(min_clusters, max_clusters), sil_score_list[1,:], '-r', label = 'Val Sil',)

            ax0.legend()
            ax00.legend()

        if predict_test_labels:
    
            best_n_cluster_bisec = 4
            params_bisec['n_clusters'] = best_n_cluster_bisec
            
            bisec = BisectingKMeans(**params_bisec)
            bisec.fit(X_train[good_features_clustering])

            # Calc acc and silhouette score
            labels = bisec.labels_
            acc, acc_arr, w_arr = clustering_accuracy(X_train[good_features_clustering], y_train, labels)
            val_labels = bisec.predict(X_val[good_features_clustering])
            val_acc,val_acc_arr,val_w_arr = clustering_accuracy(X_val[good_features_clustering], y_val, val_labels)

            sil_score = silhouette_score(X_train[good_features_clustering].values, 
                                                                labels, sample_size = 20_000)
            val_sil_score = silhouette_score(X_val[good_features_clustering].values, val_labels,)

            print(bisec.inertia_)
            print("train/val acc: ", acc, val_acc)
            print("train/val sil: ", sil_score, val_sil_score)

            print("val")
            print(val_acc_arr)
            print(val_w_arr)
            print("train")
            print(acc_arr)
            print(w_arr)

            # Predict clusters on test data 
            ## Finally, predict test data clustering labels
            bisec_test_labels = bisec.predict(scaled_test_features[good_features_clustering])
    
            #save model
            from joblib import dump, load
            dump(bisec, 'bisec.joblib')

            if 0:
                # save results to file
                with open('Clustering_SimonGuldager_SKLearn_BisectingKMeans.txt', 'w') as file:
                    # Iterate over the array elements
                    for i, value in enumerate(bisec_test_labels):
                        # Write the index and value separated by a comma
                        file.write(f'{i}, {int(value)}\n')
 

  

    plt.show()


if __name__ == '__main__':
    main()

if 0:
    def P1():
        pass

    def P2():
        pass

    def P3():
        pass

    def P4():
        pass

    def P5():
        pass

    def main():
        ## Set which problems to run
        p1, p2, p3, p4, p5 = False, False, False, False, False
        problem_numbers = [p1, p2, p3, p4, p5]
        f_list = [P1, P2, P3, P4, P5]

        for i, f in enumerate(f_list):
            if problem_numbers[i]:
                print(f'\nPROBLEM {i + 1}:')
                f()

    if __name__ == '__main__':
        main()
