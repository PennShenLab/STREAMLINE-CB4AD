"""
File:ModelJob.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 5 of STREAMLINE - This 'Job' script is called by ModelMain.py and runs machine learning modeling using respective training datasets.
            This pipeline currently includes the following 13 ML modeling algorithms for binary classification:
            * Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LGBoost, CatBoost, Support Vector Machine (SVM), Artificial Neural Network (ANN),
            * k Nearest Neighbors (k-NN), Educational Learning Classifier System (eLCS), X Classifier System (XCS), and the Extended Supervised Tracking and Classifying System (ExSTraCS)
            This phase includes hyperparameter optimization of all algorithms (other than naive bayes), model training, model feature importance estimation (using internal algorithm
            estimations, if available, or via permutation feature importance), and performance evaluation on hold out testing data. This script runs for a single combination of a
            cv dataset (for each original target dataset) and ML modeling algorithm.
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import sys
import time
import random
import pandas as pd
import numpy as np
import os
import pickle
import copy
import math
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
#Model Packages:
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import SGDClassifier
# from sklearn import tree
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ParameterGrid
# import xgboost as xgb
# import lightgbm as lgb
# import catboost as cgb
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from gplearn.genetic import SymbolicClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from group_lasso import GroupLasso
import l21regjob
#import gplearn as gp
# from skeLCS import eLCS
# from skXCS import XCS
# from skExSTraCS import ExSTraCS
#Evalutation metrics
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import balanced_accuracy_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn import metrics
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from scipy.stats import pearsonr
#Other packages
from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.inspection import permutation_importance
import optuna #hyperparameter optimization
import networkx as nx

def job(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,instance_label,class_label,random_state,cvCount,filter_poor_features,training_subsample,use_uniform_FI,primary_metric,algAbrev,jupyterRun):
    """ Specifies hardcoded (below) range of hyperparameter options selected for each ML algorithm and then runs the modeling method. Set up this way so that users can easily modify ML hyperparameter settings when running from the Jupyter Notebook. """
    if n_trials == 'None':
        n_trials = None
    else:
        n_trials = int(n_trials)
    if timeout == 'None':
        timeout = None
    else:
        timeout = int(timeout)
    #Add spaces back to algorithm names
    algorithm = algorithm.replace("_", " ")
    if eval(jupyterRun):
        print('Running '+str(algorithm)+' on '+str(train_file_path))
    #Get header names for current CV dataset for use later in GP tree visulaization
    data_name = full_path.split('/')[-1]
    feature_names = pd.read_csv(full_path+'/CVDatasets/'+data_name+'_CV_'+str(cvCount)+'_Test.csv').columns.values.tolist()
    if instance_label != 'None':
        feature_names.remove(instance_label)
    feature_names.remove(class_label)
    #Get hyperparameter grid
    param_grid = hyperparameters(random_state,do_lcs_sweep,nu,iterations,N,feature_names)[algorithm]
    runModel(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,export_hyper_sweep_plots, instance_label,class_label,random_state,cvCount,filter_poor_features,training_subsample,use_uniform_FI,primary_metric,param_grid,algAbrev)


def runModel(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,training_subsample,use_uniform_FI,primary_metric,param_grid,algAbrev):
    """ Run all elements of modeling: loading data, hyperparameter optimization, model training, and evaluation on hold out testing data.  Each ML algorithm has its own method below to handle these steps. """
    job_start_time = time.time() #for tracking phase runtime
    # Set random seeds for replicatability
    random.seed(random_state)
    np.random.seed(random_state)
    #Load training and testing datasets separating features from outcome for scikit-learn-based modeling
    
    trainX,trainY,testX,testY = dataPrep(train_file_path,instance_label,class_label,test_file_path)
    
    #Run ml modeling algorithm specified-------------------------------------------------------------------------------------------------------------------------------------------------------------
    if algorithm == 'Linear Regression':
        ret, resd = run_LiR_full(trainX,trainY,testX,testY,random_state,cvCount,param_grid,n_trials,timeout,export_hyper_sweep_plots,full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'Elastic Net':
        ret, resd = run_EN_full(trainX,trainY,testX,testY,random_state,cvCount,param_grid,n_trials,timeout,export_hyper_sweep_plots,full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'Group Lasso':
        ret, resd = run_GL_full(trainX,trainY,testX,testY,random_state,cvCount,param_grid,n_trials,timeout,export_hyper_sweep_plots,full_path,use_uniform_FI,primary_metric)
        
        
    ### Add new Regression algorithms here...
    
        
    #Pickle all evaluation metrics for ML model training and evaluation
    if not os.path.exists(full_path+'/model_evaluation/pickled_metrics'):
        os.mkdir(full_path+'/model_evaluation/pickled_metrics')
    pickle.dump(ret, open(full_path + '/model_evaluation/pickled_metrics/' + algAbrev + '_CV_' + str(cvCount) + "_metrics.pickle", 'wb'))
    pickle.dump(resd, open(full_path + '/model_evaluation/pickled_metrics/' + algAbrev + '_CV_' + str(cvCount) + "_residuals.pickle", 'wb'))
    #Save runtime of ml algorithm training and evaluation
    saveRuntime(full_path,job_start_time,algAbrev,algorithm,cvCount)
    # Print phase completion
    print(full_path.split('/')[-1] + " [CV_" + str(cvCount) + "] ("+algAbrev+") training complete. ------------------------------------")
    experiment_path = '/'.join(full_path.split('/')[:-1])
    job_file = open(experiment_path + '/jobsCompleted/job_model_' + full_path.split('/')[-1] + '_' + str(cvCount) +'_' +algAbrev+'.txt', 'w')
    job_file.write('complete')
    job_file.close()

def runModel_2(algorithm,train_file_path_1, train_file_path_2, train_file_path_3,test_file_path_1, test_file_path_2, test_file_path_3,full_path,n_trials,timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,training_subsample,use_uniform_FI,primary_metric,param_grid,algAbrev, output_path, experiment_name):
    """ Run all elements of modeling: loading data, hyperparameter optimization, model training, and evaluation on hold out testing data.  Each ML algorithm has its own method below to handle these steps. """
    job_start_time = time.time() #for tracking phase runtime
    # Set random seeds for replicatability
    random.seed(random_state)
    np.random.seed(random_state)
    #Load training and testing datasets separating features from outcome for scikit-learn-based modeling
    trainX,trainY, diag, testX,testY, ROIs, datasets = dataPrep_2(train_file_path_1, train_file_path_2, train_file_path_3, instance_label,class_label, test_file_path_1, test_file_path_2, test_file_path_3)
    
    G = nx.Graph()
    
    for i in range(len(ROIs)):
      G.add_node(ROIs[i])

    temporal_lobe = np.array([[37, 38], [39, 40], [41, 42], [55, 56], [79, 80], [81, 82], [83, 84], [85, 86], [87, 88], [89, 90]])
    temporal_lobe = np.subtract(temporal_lobe, 1)

    posterior_fossa = np.array([[91, 92],[93, 94],[95, 96],[97, 98],[99, 100],[101, 102],[103, 104],[105, 106],[107, 108]])
    posterior_fossa = np.subtract(posterior_fossa, 1)

    insula_gyri = np.array([[29, 30],[31, 32],[33, 34],[35, 36]])
    insula_gyri = np.subtract(insula_gyri, 1)

    frontal_lobe = np.array([[1, 2],[3, 4],[5, 6],[7, 8],[9, 10],[11, 12],[13, 14],[15, 16],[17, 18],[19, 20],[21, 22],[23, 24],[25, 26],[27, 28],[69, 70]])
    frontal_lobe = np.subtract(frontal_lobe, 1)

    occipital_lobe = np.array([[43, 44],[45, 46],[47, 48],[49, 50],[51, 52],[53, 54]])
    occipital_lobe = np.subtract(occipital_lobe, 1)

    parietal_lobe = np.array([[57, 58],[59, 60],[61, 62],[63, 64],[65, 66],[67, 68]])
    parietal_lobe = np.subtract(parietal_lobe, 1)

    central_structures = np.array([[53, 54], [55, 56], [57, 58], [59, 60]])
    central_structures = np.subtract(central_structures, 1)

    whole_connections = np.concatenate((temporal_lobe, posterior_fossa, insula_gyri, frontal_lobe, occipital_lobe, parietal_lobe, central_structures))

    for connection in whole_connections:
      G.add_edge(ROIs[connection[0]], ROIs[connection[1]])
    
    #Run ml modeling algorithm specified-------------------------------------------------------------------------------------------------------------------------------------------------------------
    if algorithm == 'L21Reg':
        ret_1, ret_2,ret_3, resd_1,  resd_2, resd_3 = run_L21_full(trainX, trainY,testX, testY, random_state, cvCount,param_grid, n_trials,timeout,export_hyper_sweep_plots,full_path,use_uniform_FI,primary_metric, ROIs, datasets)
    elif algorithm == 'L21GMMReg':
        ret_1, ret_2,ret_3, resd_1,  resd_2, resd_3 = run_L21GMM_full(trainX, trainY,testX, testY, G, random_state, cvCount,param_grid, n_trials,timeout,export_hyper_sweep_plots,full_path,use_uniform_FI,primary_metric, ROIs, datasets)
    elif algorithm == 'L21DGMMReg':
        ret_1, ret_2,ret_3, resd_1,  resd_2, resd_3 = run_L21DGMM_full(trainX, trainY,testX, testY, diag, random_state, cvCount,param_grid, n_trials,timeout,export_hyper_sweep_plots,full_path,use_uniform_FI,primary_metric, ROIs, datasets)
    
    #Pickle all evaluation metrics for ML model training and evaluation
    if not os.path.exists(full_path+'/model_evaluation'):
        os.mkdir(full_path+'/model_evaluation')
    if not os.path.exists(full_path+'/model_evaluation/pickled_metrics'):
        os.mkdir(full_path+'/model_evaluation/pickled_metrics')
    for i in range(len(datasets)):
        if not os.path.exists(output_path+'/'+experiment_name+'/'+datasets[i]+'/model_evaluation'):
            os.mkdir(output_path+'/'+experiment_name+'/'+datasets[i]+'/model_evaluation')
        if not os.path.exists(output_path+'/'+experiment_name+'/'+datasets[i]+'/model_evaluation/pickled_metrics'):
            os.mkdir(output_path+'/'+experiment_name+'/'+datasets[i]+'/model_evaluation/pickled_metrics')

    pickle.dump(ret_1, open(output_path+'/'+experiment_name+'/'+datasets[0] + '/model_evaluation/pickled_metrics/' + algAbrev + '_CV_' + str(cvCount) + "_metrics.pickle", 'wb'))
    pickle.dump(ret_2, open(output_path+'/'+experiment_name+'/'+datasets[1] + '/model_evaluation/pickled_metrics/' + algAbrev + '_CV_' + str(cvCount) + "_metrics.pickle", 'wb'))
    pickle.dump(ret_3, open(output_path+'/'+experiment_name+'/'+datasets[2] + '/model_evaluation/pickled_metrics/' + algAbrev + '_CV_' + str(cvCount) + "_metrics.pickle", 'wb'))
    pickle.dump(resd_1, open(output_path+'/'+experiment_name+'/'+datasets[0] + '/model_evaluation/pickled_metrics/' + algAbrev + '_CV_' + str(cvCount) + "_residuals.pickle", 'wb'))
    pickle.dump(resd_2, open(output_path+'/'+experiment_name+'/'+datasets[1] + '/model_evaluation/pickled_metrics/' + algAbrev + '_CV_' + str(cvCount) + "_residuals.pickle", 'wb'))
    pickle.dump(resd_3, open(output_path+'/'+experiment_name+'/'+datasets[2] + '/model_evaluation/pickled_metrics/' + algAbrev + '_CV_' + str(cvCount) + "_residuals.pickle", 'wb'))
    #Save runtime of ml algorithm training and evaluation
    if not os.path.exists(full_path+'/runtime'):
        os.mkdir(full_path+'/runtime')
    saveRuntime(output_path+'/'+experiment_name+'/'+datasets[0],job_start_time,algAbrev,algorithm,cvCount)
    saveRuntime(output_path+'/'+experiment_name+'/'+datasets[1],job_start_time,algAbrev,algorithm,cvCount)
    saveRuntime(output_path+'/'+experiment_name+'/'+datasets[2],job_start_time,algAbrev,algorithm,cvCount)
    # Print phase completion
    print(full_path.split('/')[-1] + " [CV_" + str(cvCount) + "] ("+algAbrev+") training complete. ------------------------------------")
    experiment_path = '/'.join(full_path.split('/')[:-1])
    job_file = open(experiment_path + '/jobsCompleted/job_model_' + full_path.split('/')[-1] + '_' + str(cvCount) +'_' +algAbrev+'.txt', 'w')
    job_file.write('complete')
    job_file.close()

def dataPrep(train_file_path,instance_label,class_label,test_file_path):
    """ Loads target cv training dataset, separates class from features and removes instance labels."""
    train = pd.read_csv(train_file_path)
    if instance_label != 'None':
        trainY = train[instance_label].values
        train = train.drop(instance_label,axis=1)
    trainX = train.drop(class_label,axis=1).values
    del train #memory cleanup
    test = pd.read_csv(test_file_path)
    if instance_label != 'None':
        testY = test[instance_label].values
        test = test.drop(instance_label,axis=1)
    testX = test.drop(class_label,axis=1).values
    del test #memory cleanup
    return trainX,trainY,testX,testY

def dataPrep_2(train_file_path_1, train_file_path_2, train_file_path_3, instance_label,class_label, test_file_path_1, test_file_path_2, test_file_path_3):
    train_1, train_2, train_3 = pd.read_csv(train_file_path_1), pd.read_csv(train_file_path_2), pd.read_csv(train_file_path_3)
    ROIs = []
    for ROI in train_1.columns:
        ROIs.append(ROI)
    ROIs.remove(instance_label)
    ROIs.remove(class_label)
    dataset_1 = train_file_path_1.split('/')[8]
    dataset_2 = train_file_path_2.split('/')[8]
    dataset_3 = train_file_path_3.split('/')[8]
    datasets = [dataset_1, dataset_2, dataset_3]
    if instance_label != 'None':
        trainY = train_1[instance_label].values
        train_1 = train_1.drop(instance_label,axis=1)
        train_2 = train_2.drop(instance_label,axis=1)
        train_3 = train_3.drop(instance_label,axis=1)
    diag = train_1[class_label].values
    trainX_1 = train_1.drop(class_label,axis=1).values
    trainX_2 = train_2.drop(class_label,axis=1).values
    trainX_3 = train_3.drop(class_label,axis=1).values
    trainX = np.array([trainX_1, trainX_2, trainX_3])
    del train_1, train_2, train_3
    test_1, test_2, test_3 = pd.read_csv(test_file_path_1), pd.read_csv(test_file_path_2), pd.read_csv(test_file_path_3)
    if instance_label != 'None':
        testY = test_1[instance_label].values
        test_1 = test_1.drop(instance_label,axis=1)
        test_2 = test_2.drop(instance_label,axis=1)
        test_3 = test_3.drop(instance_label,axis=1)
    testX_1 = test_1.drop(class_label,axis=1).values
    testX_2 = test_2.drop(class_label,axis=1).values
    testX_3 = test_3.drop(class_label,axis=1).values
    testX = np.array([testX_1, testX_2, testX_3])
    del test_1, test_2, test_3
    return trainX, trainY, diag, testX, testY, ROIs, datasets
    
def saveRuntime(full_path,job_start_time,algAbrev,algorithm,cvCount):
    """ Save ML algorithm training and evaluation runtime for this phase."""
    runtime_file = open(full_path + '/runtime/runtime_'+algAbrev+'_CV'+str(cvCount)+'.txt', 'w')
    runtime_file.write(str(time.time() - job_start_time))
    runtime_file.close()


def hyper_eval(est, x_train, y_train, random_state, hype_cv, params, scoring_metric):
    """ Run hyperparameter evaluation for a given ML algorithm using Optuna. Uses further k-fold cv within target training data for hyperparameter evaluation."""
    cv = StratifiedKFold(n_splits=hype_cv, shuffle=True, random_state=random_state)
    model = clone(est).set_params(**params)
    #Flexibly handle whether random seed is given as 'random_seed' or 'seed' - scikit learn uses 'random_seed'
    #for a in ['random_state','seed']:
    #    if hasattr(model,a):
    #        setattr(model,a,random_state)
    performance = np.mean(cross_val_score(model,x_train,y_train,cv=cv,scoring=scoring_metric,verbose=0))
    return performance

def hyper_eval_Regression(est, x_train, y_train, random_state, hype_cv, params, scoring_metric):
    """ Run hyperparameter evaluation for a given ML algorithm using Optuna. Uses further k-fold cv within target training data for hyperparameter evaluation."""
    cv = KFold(n_splits=hype_cv, shuffle=True, random_state=random_state)
    model = clone(est).set_params(**params)
    #Flexibly handle whether random seed is given as 'random_seed' or 'seed' - scikit learn uses 'random_seed'
    performance = np.mean(cross_val_score(model,x_train,y_train,cv=cv,scoring=scoring_metric,verbose=0))
    return performance

def hyper_eval_2(est, x_train, y_train, random_state, hype_cv, params, scoring_metric, G = None, diag = np.full(1, np.nan)):
    """ Run hyperparameter evaluation for a given ML algorithm using Optuna. Uses further k-fold cv within target training data for hyperparameter evaluation."""
    cv = KFold(n_splits=hype_cv, shuffle=True, random_state=random_state)
    model = clone(est).set_params(**params)
    #Flexibly handle whether random seed is given as 'random_seed' or 'seed' - scikit learn uses 'random_seed'
    performance = np.mean(cross_val_score_2(model,x_train[0], x_train[1], x_train[2] ,y_train, G = G, diag = diag, cv=cv,scoring=scoring_metric))
    return performance

def cross_val_score_2(est,X_1, X_2, X_3,y, G, diag, cv, scoring):
    scores = []
    if scoring == 'explained_variance':
        for train_index, test_index in cv.split(X_1, y):
            X_train_1, X_test_1 = X_1[train_index],X_1[test_index]
            X_train_2, X_test_2 = X_2[train_index],X_2[test_index]
            X_train_3, X_test_3 = X_3[train_index],X_3[test_index]
            y_train, y_test = y[train_index],y[test_index]
            X_train = np.array([X_train_1, X_train_2, X_train_3])
            X_test = np.array([X_test_1, X_test_2, X_test_3])
            if (G == None) and all(np.isnan(diag)):
                est.fit(X_train, y_train)
            elif (G != None) and all(np.isnan(diag)):
                est.fitG(X_train, y_train, G)
            elif (G == None) and (all(np.isnan(diag)) == False):
                diag_train = diag[train_index]
                est.fitD(X_train, y_train, diag_train)
            else:
                raise ValueError("The indicated fitting strategy does not exist. Please attemp from L21, L21GMM, and L21DGMM.")
            y_pred = est.predict(X_test)
            evs_1 = explained_variance_score(np.squeeze(y_test),y_pred[0])
            evs_2 = explained_variance_score(np.squeeze(y_test),y_pred[1])
            evs_3 = explained_variance_score(np.squeeze(y_test),y_pred[2])
            score = np.mean(np.array([evs_1, evs_2, evs_3]))
            scores.append(score)
    elif scoring == 'max_error':
        for train_index, test_index in cv.split(X_1, y):
            X_train_1, X_test_1 = X_1[train_index],X_1[test_index]
            X_train_2, X_test_2 = X_2[train_index],X_2[test_index]
            X_train_3, X_test_3 = X_3[train_index],X_3[test_index]
            y_train, y_test = y[train_index],y[test_index]
            X_train = np.array([X_train_1, X_train_2, X_train_3])
            X_test = np.array([X_test_1, X_test_2, X_test_3])
            if (G == None) and (diag == None).all():
                est.fit(X_train, y_train)
            elif (G != None) and (diag == None).all():
                est.fitG(X_train, y_train, G)
            elif (G == None) and (diag != None).all():
                diag_train = diag[train_index]
                est.fitD(X_train, y_train, diag_train)
            else:
                raise ValueError("The indicated fitting strategy does not exist. Please attemp from L21, L21GMM, and L21DGMM.")
            y_pred = est.predict(X_test)
            me_1 = max_error(np.squeeze(y_test), y_pred[0])
            me_2 = max_error(np.squeeze(y_test), y_pred[1])
            me_3 = max_error(np.squeeze(y_test), y_pred[2])
            score = np.max(np.array([me_1, me_2, me_3]))
            scores.append(score)
    elif scoring == 'neg_mean_absolute_error':
        for train_index, test_index in cv.split(X_1, y):
            X_train_1, X_test_1 = X_1[train_index],X_1[test_index]
            X_train_2, X_test_2 = X_2[train_index],X_2[test_index]
            X_train_3, X_test_3 = X_3[train_index],X_3[test_index]
            y_train, y_test = y[train_index],y[test_index]
            X_train = np.array([X_train_1, X_train_2, X_train_3])
            X_test = np.array([X_test_1, X_test_2, X_test_3])
            if (G == None) and (diag == None).all():
                est.fit(X_train, y_train)
            elif (G != None) and (diag == None).all():
                est.fitG(X_train, y_train, G)
            elif (G == None) and (diag != None).all():
                diag_train = diag[train_index]
                est.fitD(X_train, y_train, diag_train)
            else:
                raise ValueError("The indicated fitting strategy does not exist. Please attemp from L21, L21GMM, and L21DGMM.")
            y_pred = est.predict(X_test)
            mae_1 = mean_absolute_error(np.squeeze(y_test), y_pred[0])
            mae_2 = mean_absolute_error(np.squeeze(y_test), y_pred[1])
            mae_3 = mean_absolute_error(np.squeeze(y_test), y_pred[2])
            score = np.mean(np.array([mae_1, mae_2, mae_3]))
            scores.append(score)
    elif scoring == 'neg_mean_squared_error':
        for train_index, test_index in cv.split(X_1, y):
            X_train_1, X_test_1 = X_1[train_index],X_1[test_index]
            X_train_2, X_test_2 = X_2[train_index],X_2[test_index]
            X_train_3, X_test_3 = X_3[train_index],X_3[test_index]
            y_train, y_test = y[train_index],y[test_index]
            X_train = np.array([X_train_1, X_train_2, X_train_3])
            X_test = np.array([X_test_1, X_test_2, X_test_3])
            if (G == None) and (diag == None).all():
                est.fit(X_train, y_train)
            elif (G != None) and (diag == None).all():
                est.fitG(X_train, y_train, G)
            elif (G == None) and (diag != None).all():
                diag_train = diag[train_index]
                est.fitD(X_train, y_train, diag_train)
            else:
                raise ValueError("The indicated fitting strategy does not exist. Please attemp from L21, L21GMM, and L21DGMM.")
            y_pred = est.predict(X_test)
            mse_1 = mean_squared_error(np.squeeze(y_test), y_pred[0])
            mse_2 = mean_squared_error(np.squeeze(y_test), y_pred[1])
            mse_3 = mean_squared_error(np.squeeze(y_test), y_pred[2])
            score = np.mean(np.array([mse_1, mse_2, mse_3]))
            scores.append(score)
    elif scoring == 'neg_median_absolute_error':
        for train_index, test_index in cv.split(X_1, y):
            X_train_1, X_test_1 = X_1[train_index],X_1[test_index]
            X_train_2, X_test_2 = X_2[train_index],X_2[test_index]
            X_train_3, X_test_3 = X_3[train_index],X_3[test_index]
            y_train, y_test = y[train_index],y[test_index]
            X_train = np.array([X_train_1, X_train_2, X_train_3])
            X_test = np.array([X_test_1, X_test_2, X_test_3])
            if (G == None) and (diag == None).all():
                est.fit(X_train, y_train)
            elif (G != None) and (diag == None).all():
                est.fitG(X_train, y_train, G)
            elif (G == None) and (diag != None).all():
                diag_train = diag[train_index]
                est.fitD(X_train, y_train, diag_train)
            else:
                raise ValueError("The indicated fitting strategy does not exist. Please attemp from L21, L21GMM, and L21DGMM.")
            y_pred = est.predict(X_test)
            mdae_1 = median_absolute_error(np.squeeze(y_test), y_pred[0])
            mdae_2 = median_absolute_error(np.squeeze(y_test), y_pred[1])
            mdae_3 = median_absolute_error(np.squeeze(y_test), y_pred[2])
            score = np.mean(np.array([mdae_1, mdae_2, mdae_3]))
            scores.append(score)
    return np.array(scores)
def residualRecordReg(clf, model, x_train, y_train, x_test,y_test):
    y_train_pred = model.predict(x_train)
    y_pred = model.predict(x_test)
    residual_train = y_train - y_train_pred
    residual_test = y_test - y_pred
    return residual_train, residual_test, y_train_pred, y_pred

def residualRecordL21Reg(clf, model, x_train, y_train, x_test,y_test):
    y_train_pred = model.predict(x_train)
    y_pred = model.predict(x_test)
    residual_train_1 = y_train - y_train_pred[0]
    residual_test_1 = y_test - y_pred[0]
    residual_train_2 = y_train - y_train_pred[1]
    residual_test_2 = y_test - y_pred[1]
    residual_train_3 = y_train - y_train_pred[2]
    residual_test_3 = y_test - y_pred[2]
    return residual_train_1, residual_test_1, residual_train_2, residual_test_2, residual_train_3, residual_test_3,y_train_pred, y_pred
    
def modelEvaluationL21Reg(clf, model, x_test,y_test):
    y_pred = model.predict(x_test)
    # y_pred = np.maximum(0.01,y_pred)
    # y_test = np.maximum(0.01,y_test)
    
    me_1 = max_error(np.squeeze(y_test), y_pred[0])
    me_2 = max_error(np.squeeze(y_test), y_pred[1])
    me_3 = max_error(np.squeeze(y_test), y_pred[2])
    me = np.max(np.array([me_1, me_2, me_3]))
    
    mae_1 = mean_absolute_error(np.squeeze(y_test), y_pred[0])
    mae_2 = mean_absolute_error(np.squeeze(y_test), y_pred[1])
    mae_3 = mean_absolute_error(np.squeeze(y_test), y_pred[2])
    mae = np.mean(np.array([mae_1, mae_2, mae_3]))
    
    mse_1 = mean_squared_error(np.squeeze(y_test), y_pred[0])
    mse_2 = mean_squared_error(np.squeeze(y_test), y_pred[1])
    mse_3 = mean_squared_error(np.squeeze(y_test), y_pred[2])
    mse = np.mean(np.array([mse_1, mse_2, mse_3]))
    
    mdae_1 = median_absolute_error(np.squeeze(y_test), y_pred[0])
    mdae_2 = median_absolute_error(np.squeeze(y_test), y_pred[1])
    mdae_3 = median_absolute_error(np.squeeze(y_test), y_pred[2])
    mdae = np.mean(np.array([mdae_1, mdae_2, mdae_3]))
    
    evs_1 = explained_variance_score(np.squeeze(y_test), y_pred[0])
    evs_2 = explained_variance_score(np.squeeze(y_test), y_pred[1])
    evs_3 = explained_variance_score(np.squeeze(y_test), y_pred[2])
    evs = np.mean(np.array([evs_1, evs_2, evs_3]))
    
    corr1 = pearsonr(np.squeeze(y_test),y_pred[0])[0]
    corr2 = pearsonr(np.squeeze(y_test),y_pred[1])[0]
    corr3 = pearsonr(np.squeeze(y_test),y_pred[2])[0]
    p_corr = np.mean(np.array([corr1, corr2, corr3]))
    
    print("Testing Max Error is: ", me)
    print("Testing Mean Absolute Error is: ", mae)
    print("Testing Mean Squared Error is: ", mse)
    print("Testing Median Absolute Error is: ", mdae)
    print("Testing Explained Variance Score is: ", evs)
    print("Testing Pearson Correlation is: ", p_corr)
    return [me_1, mae_1, mse_1, mdae_1, evs_1, corr1], [me_2, mae_2,mse_2, mdae_2, evs_2, corr2], [me_3, mae_3, mse_3, mdae_3, evs_3, corr3]
    
def modelEvaluation_Regression(clf,model,x_test,y_test):
    """ Runs commands to gather all evaluations for later summaries and plots. """
    #Prediction evaluation
    y_pred = model.predict(x_test)
    metricList = regressionEval(y_test, y_pred)
    return metricList
    
def save_fig(path, name, tight_layout=True, resolution=300):
    print("Saving figure", name)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=resolution)
    
def heatMap(clf, model, i, name, ROIs, datasets, full_path):
    weight = model.coef_
    df_weight =  pd.DataFrame(weight,columns=datasets)
    plt.figure(figsize=(10,15))
    ax = sns.heatmap(df_weight, yticklabels=ROIs)
    ax.set(xlabel='Modality', ylabel='ROIs')
    path = full_path + '/models/' + name + '_' + 'CV' + str(i)+ '_heatmap.png'
    save_fig(path, name + '_' + 'CV' + str(i))  

#Elastic Net #############################################################################################################################
def objective_EN(trial, est, x_train, y_train, random_state, hype_cv, param_grid, scoring_metric):
    """ Prepares Elastic Net hyperparameter variables for Optuna run hyperparameter optimization. """
    params = {'alpha': trial.suggest_loguniform('alpha', param_grid['alpha'][0], param_grid['alpha'][1]),
              'l1_ratio' : trial.suggest_uniform('l1_ratio',param_grid['l1_ratio'][0], param_grid['l1_ratio'][1]),
              'max_iter' : trial.suggest_uniform('max_iter',param_grid['max_iter'][0], param_grid['max_iter'][1]),
              'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval_Regression(est, x_train, y_train, random_state, hype_cv, params, scoring_metric)

def run_EN_full(x_train, y_train, x_test, y_test,random_state,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    """ Run Elastic Net hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Specify algorithm for hyperparameter optimization
    est = ElasticNet()

    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=random_state)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_EN(trial, est, x_train, y_train, random_state, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            try:
                fig = optuna.visualization.plot_parallel_coordinate(study)
                fig.write_image(full_path+'/models/EN_ParamOptimization_'+str(i)+'.png')
            except:
                print('Warning: Optuna Optimization Visualization Generation Failed for DT Due to Known Release Issue.  Please install Optuna 2.0.0 to avoid this issue.')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        est = ElasticNet()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/models/EN_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/models/EN_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    print('weights:', model.get_params())
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/EN_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList = modelEvaluation_Regression(clf,model,x_test,y_test)
    # Residual Calculation
    residual_train, residual_test, y_train_pred, y_pred = residualRecordReg(clf, model, x_train, y_train, x_test,y_test)
    #Feature Importance Estimates
    results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=random_state)
    fi = results.importances_mean
    return [metricList, fi], [residual_train, residual_test, y_train_pred, y_pred, y_train, y_test]
    
#Group Lasso #############################################################################################################################
def objective_GL(trial, est, x_train, y_train, random_state, hype_cv, param_grid, scoring_metric):
    """ Prepares Group Lasso hyperparameter variables for Optuna run hyperparameter optimization. """
    params = {'group_reg': trial.suggest_loguniform('group_reg', param_grid['group_reg'][0], param_grid['group_reg'][1]),
              #'l1_reg' : trial.suggest_uniform('l1_reg',param_grid['l1_reg'][0], param_grid['l1_reg'][1]),
              'n_iter' : trial.suggest_int('n_iter',param_grid['n_iter'][0], param_grid['n_iter'][1]),
              'scale_reg' : trial.suggest_categorical('scale_reg',param_grid['scale_reg']),
              #'subsampling_scheme' : trial.suggest_uniform('subsampling_scheme',param_grid['subsampling_scheme'][0], param_grid['subsampling_scheme'][1]),
              #'frobenius_lipschitz' : trial.suggest_categorical('frobenius_lipschitz',param_grid['frobenius_lipschitz']),
              'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval_Regression(est, x_train, y_train, random_state, hype_cv, params, scoring_metric)

def run_GL_full(x_train, y_train, x_test, y_test,random_state,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    """ Run Group Lasso hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False

    #Read the file with group information
    try:
        groups_path = '/content/drive/MyDrive/STREAMLINE-Regression/streamline/groups.csv'
        groups = pd.read_csv(groups_path, header=None)
    except:
        print('File with group information not found, or invalid extension, or inappropriate format')
    
    #Specify algorithm for hyperparameter optimization
    gl = GroupLasso(groups, l1_reg=0, supress_warning=True)

    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=random_state)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_GL(trial, gl, x_train, y_train, random_state, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            try:
                fig = optuna.visualization.plot_parallel_coordinate(study)
                fig.write_image(full_path+'/models/GL_ParamOptimization_'+str(i)+'.png')
            except:
                print('Warning: Optuna Optimization Visualization Generation Failed for DT Due to Known Release Issue.  Please install Optuna 2.0.0 to avoid this issue.')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        gl = GroupLasso(groups, l1_reg=0, supress_warning=True)
        clf = gl.set_params(**best_trial.params)
        export_best_params(full_path + '/models/GL_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = gl.set_params(**params)
        export_best_params(full_path + '/models/GL_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    #print('weights:', model.get_params())
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/GL_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList = modelEvaluation_Regression(clf,model,x_test,y_test)
    # Residual Calculation
    residual_train, residual_test, y_train_pred, y_pred = residualRecordReg(clf, model, x_train, y_train, x_test,y_test)
    #Feature Importance Estimates
    results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=random_state)
    fi = results.importances_mean
    return [metricList, fi], [residual_train, residual_test, y_train_pred, y_pred, y_train, y_test]

#Linear Regression #############################################################################################################################
def run_LiR_full(x_train, y_train, x_test, y_test,random_state,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    """ Run Linear Regression model training, evaluation, and model feature importance estimation. No hyperparameters to optimize."""
    #Train model using 'best' hyperparameters - Uses default 3-fold internal CV (training/validation splits)
    clf = LinearRegression()
    model = clf.fit(x_train, y_train)
    print('weights:', model.get_params())
    #Save model
    pickle.dump(model, open(full_path+'/models/pickledModels/LiR_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList = modelEvaluation_Regression(clf,model,x_test,y_test)
    # Residual Calculation
    residual_train, residual_test, y_train_pred, y_pred = residualRecordReg(clf, model, x_train, y_train, x_test,y_test)
    #Feature Importance Estimates
    results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=random_state)
    fi = results.importances_mean
    return [metricList, fi], [residual_train, residual_test, y_train_pred, y_pred, y_train, y_test]
    


#L21 Regression ####################################################################################################################################
def objective_L21(trial, est, x_train, y_train, random_state, hype_cv, param_grid, scoring_metric):
    params = {'lambda1' : trial.suggest_loguniform('lambda1',param_grid['lambda1'][0], param_grid['lambda1'][1]),
			  #'lambda2' : trial.suggest_loguniform('lambda2',param_grid['lambda2'][0], param_grid['lambda2'][1]),
			  'max_iter' : trial.suggest_loguniform('max_iter',param_grid['max_iter'][0], param_grid['max_iter'][1])}

    return hyper_eval_2(est, x_train, y_train, random_state, hype_cv, params, scoring_metric)

def run_L21_full(x_train, y_train, x_test, y_test, random_state, i,param_grid, n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric, ROIs, datasets):
    """ Run L21MM hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Specify algorithm for hyperparameter optimization
    est = l21regjob.L21Reg()
    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=random_state)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_L21(trial, est, x_train, y_train, random_state, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #study.best_trial.user_attrs['params']
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            try:
                fig = optuna.visualization.plot_parallel_coordinate(study)
                fig.write_image(full_path+'/models/L21_ParamOptimization_'+str(i)+'.png')
            except:
                print('Warning: Optuna Optimization Visualization Generation Failed for L21 Due to Known Release Issue.  Please install Optuna 2.0.0 to avoid this issue.')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        est = l21regjob.L21Reg()
        clf = est.set_params(**best_trial.params)
        if not os.path.exists(full_path + '/models'):
            os.mkdir(full_path + '/models')
        export_best_params(full_path + '/models/L21_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        if not os.path.exists(full_path + '/models'):
            os.mkdir(full_path + '/models')
        export_best_params(full_path + '/models/L21_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    #print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    # Save model with pickle so it can be applied in the future
    if not os.path.exists(full_path+'/models/pickledModels'):
        os.mkdir(full_path + '/models/pickledModels')
    pickle.dump(model, open(full_path+'/models/pickledModels/L21_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList_1,metricList_2, metricList_3 = modelEvaluationL21Reg(clf,model,x_test,y_test)
    # Residual Record
    residual_train_1, residual_test_1, residual_train_2, residual_test_2, residual_train_3, residual_test_3,y_train_pred, y_pred = residualRecordL21Reg(clf, model, x_train, y_train, x_test, y_test)
    # Feature Importance Estimates
    if eval(use_uniform_FI):
        model.idxPred = 0
        results = permutation_importance(model, x_train[0], y_train, n_repeats=10,random_state=random_state, scoring= 'explained_variance')
        fi_1 = results.importances_mean
        
        model.idxPred = 1
        results = permutation_importance(model, x_train[1], y_train, n_repeats=10,random_state=random_state, scoring= 'explained_variance')
        fi_2 = results.importances_mean
        
        model.idxPred = 2
        results = permutation_importance(model, x_train[2], y_train, n_repeats=10,random_state=random_state, scoring= 'explained_variance')
        fi_3 = results.importances_mean
    else:
        fi_1 = pow(math.e,model.coef_[0])
        fi_2 = pow(math.e,model.coef_[0])
        fi_3 = pow(math.e,model.coef_[0])
    heatMap(clf, model, i, 'L21Reg', ROIs, datasets, full_path)
    return [metricList_1, fi_1], [metricList_2, fi_2], [metricList_3, fi_3], [residual_train_1, residual_test_1, y_train_pred[0], y_pred[0], y_train, y_test], [residual_train_2, residual_test_2, y_train_pred[1], y_pred[1], y_train, y_test], [residual_train_3, residual_test_3, y_train_pred[2], y_pred[2], y_train, y_test]
    
#L21GMM Regression #################################################################################################################################
def objective_L21GMM(trial, est, x_train, y_train, G, random_state, hype_cv, param_grid, scoring_metric):
    params = {'lambda1' : trial.suggest_loguniform('lambda1',param_grid['lambda1'][0], param_grid['lambda1'][1]),
			  'lambda2' : trial.suggest_loguniform('lambda2',param_grid['lambda2'][0], param_grid['lambda2'][1]),
			  'max_iter' : trial.suggest_loguniform('max_iter',param_grid['max_iter'][0], param_grid['max_iter'][1])}

    return hyper_eval_2(est, x_train, y_train, random_state, hype_cv, params, scoring_metric, G = G)
    
def run_L21GMM_full(x_train, y_train, x_test, y_test, G, random_state, i,param_grid, n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric, ROIs, datasets):
    """ Run L21GMM hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Specify algorithm for hyperparameter optimization
    est = l21regjob.L21Reg()
    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=random_state)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_L21GMM(trial, est, x_train, y_train, G, random_state, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #study.best_trial.user_attrs['params']
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            try:
                fig = optuna.visualization.plot_parallel_coordinate(study)
                fig.write_image(full_path+'/models/L21GMM_ParamOptimization_'+str(i)+'.png')
            except:
                print('Warning: Optuna Optimization Visualization Generation Failed for L21GMM Due to Known Release Issue.  Please install Optuna 2.0.0 to avoid this issue.')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        est = l21regjob.L21Reg()
        clf = est.set_params(**best_trial.params)
        if not os.path.exists(full_path + '/models'):
            os.mkdir(full_path + '/models')
        export_best_params(full_path + '/models/L21GMM_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        if not os.path.exists(full_path + '/models'):
            os.mkdir(full_path + '/models')
        export_best_params(full_path + '/models/L21GMM_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    #print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fitG(x_train, y_train, G)
    # Save model with pickle so it can be applied in the future
    if not os.path.exists(full_path+'/models/pickledModels'):
        os.mkdir(full_path + '/models/pickledModels')
    pickle.dump(model, open(full_path+'/models/pickledModels/L21GMM_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList_1,metricList_2, metricList_3 = modelEvaluationL21Reg(clf,model,x_test,y_test)
    # Feature Importance Estimates
    if eval(use_uniform_FI):
        model.idxPred = 0
        results = permutation_importance(model, x_train[0], y_train, n_repeats=10,random_state=random_state, scoring= 'explained_variance')
        fi_1 = results.importances_mean
        
        model.idxPred = 1
        results = permutation_importance(model, x_train[1], y_train, n_repeats=10,random_state=random_state, scoring= 'explained_variance')
        fi_2 = results.importances_mean
        
        model.idxPred = 2
        results = permutation_importance(model, x_train[2], y_train, n_repeats=10,random_state=random_state, scoring= 'explained_variance')
        fi_3 = results.importances_mean
    else:
        fi_1 = pow(math.e,model.coef_[0])
        fi_2 = pow(math.e,model.coef_[0])
        fi_3 = pow(math.e,model.coef_[0])
    heatMap(clf, model, i, 'L21GMMReg', ROIs, datasets, full_path)
    return [metricList_1, fi_1], [metricList_2, fi_2], [metricList_3, fi_3]
    
#L21DGMM Regression ################################################################################################################################
def objective_L21DGMM(trial, est, x_train, y_train, diag, random_state, hype_cv, param_grid, scoring_metric):
    params = {'lambda1' : trial.suggest_loguniform('lambda1',param_grid['lambda1'][0], param_grid['lambda1'][1]),
			  'lambda2' : trial.suggest_loguniform('lambda2',param_grid['lambda2'][0], param_grid['lambda2'][1]),
			  'max_iter' : trial.suggest_loguniform('max_iter',param_grid['max_iter'][0], param_grid['max_iter'][1])}
	
    return hyper_eval_2(est, x_train, y_train, random_state, hype_cv, params, scoring_metric, diag = diag)

def run_L21DGMM_full(x_train, y_train, x_test, y_test, diag, random_state, i,param_grid, n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric, ROIs, datasets):
    """ Run L21DGMM hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Specify algorithm for hyperparameter optimization
    est = l21regjob.L21Reg()
    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=random_state)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_L21DGMM(trial, est, x_train, y_train, diag, random_state, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #study.best_trial.user_attrs['params']
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            try:
                fig = optuna.visualization.plot_parallel_coordinate(study)
                fig.write_image(full_path+'/models/L21DGMM_ParamOptimization_'+str(i)+'.png')
            except:
                print('Warning: Optuna Optimization Visualization Generation Failed for L21DGMM Due to Known Release Issue.  Please install Optuna 2.0.0 to avoid this issue.')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        est = l21regjob.L21Reg()
        clf = est.set_params(**best_trial.params)
        if not os.path.exists(full_path + '/models'):
            os.mkdir(full_path + '/models')
        export_best_params(full_path + '/models/L21DGMM_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        if not os.path.exists(full_path + '/models'):
            os.mkdir(full_path + '/models')
        export_best_params(full_path + '/models/L21DGMM_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    #print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fitD(x_train, y_train, diag)
    # Save model with pickle so it can be applied in the future
    if not os.path.exists(full_path+'/models/pickledModels'):
        os.mkdir(full_path + '/models/pickledModels')
    pickle.dump(model, open(full_path+'/models/pickledModels/L21DGMM_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList_1,metricList_2, metricList_3 = modelEvaluationL21Reg(clf,model,x_test,y_test)
    # Feature Importance Estimates
    if eval(use_uniform_FI):
        model.idxPred = 0
        results = permutation_importance(model, x_train[0], y_train, n_repeats=10,random_state=random_state, scoring= 'explained_variance')
        fi_1 = results.importances_mean
        
        model.idxPred = 1
        results = permutation_importance(model, x_train[1], y_train, n_repeats=10,random_state=random_state, scoring= 'explained_variance')
        fi_2 = results.importances_mean
        
        model.idxPred = 2
        results = permutation_importance(model, x_train[2], y_train, n_repeats=10,random_state=random_state, scoring= 'explained_variance')
        fi_3 = results.importances_mean
    else:
        fi_1 = pow(math.e,model.coef_[0])
        fi_2 = pow(math.e,model.coef_[0])
        fi_3 = pow(math.e,model.coef_[0])
    heatMap(clf, model, i, 'L21DGMMReg', ROIs, datasets, full_path)
    return [metricList_1, fi_1], [metricList_2, fi_2], [metricList_3, fi_3]


def export_best_params(file_name,param_grid):
    """ Exports best hyperparameter scores to output file."""
    best_params_copy = param_grid
    for best in best_params_copy:
        best_params_copy[best] = [best_params_copy[best]]
    df = pd.DataFrame.from_dict(best_params_copy)
    df.to_csv(file_name, index=False)


def regressionEval(y_true, y_pred):
    """ Calculates standard regression metrics including:
    max error, mean absolute error, mean squared error, r2 score, mean poisson deviance, mean gamma deviance, mean d2 tweedie score"""
    
    y_pred = np.maximum(0.01,y_pred)
    y_true = np.maximum(0.01,y_true)
    #Calculate max error.
    me = max_error(y_true, y_pred)
    #Calculate mean absolute error.
    mae = mean_absolute_error(y_true, y_pred)
    #Calculate mean squared error.
    mse = mean_squared_error(y_true, y_pred)
    #Calculate median absolute error
    mdae = median_absolute_error(y_true, y_pred)
    #Calculate explained variance score
    evs = explained_variance_score(y_true, y_pred)
    #Calculate pearson correlation
    p_corr = pearsonr(y_true, y_pred)[0]
    
    return [me, mae, mse, mdae, evs, p_corr]

def hyperparameters(random_state,feature_names):
    param_grid = {}

    param_grid_L21Reg = {'lambda1':[1,100], 'max_iter': [10, 2500]}

    param_grid_L21GMMReg = {'lambda1':[1, 300], 'lambda2':[1, 300], 'max_iter': [10, 2500]}

    param_grid_L21DGMMReg = {'lambda1': [0.01, 100], 'lambda2': [0.01, 100], 'max_iter': [10, 2500]}

    # Elastic Net Regressor
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
    param_grid_EN = {'alpha':[1e-3,1],'max_iter': [10,2500],                      
                     'l1_ratio':[0,1],'random_state':[random_state]}
    # Group Lasso Regressor
    # https://group-lasso.readthedocs.io/en/latest/api_reference.html#
    param_grid_GL = {'group_reg':[1e-3,1],#'l1_reg':[0,1],
                     'n_iter':[10,2500],
                     'scale_reg': ['group_size', 'none', 'inverse_group_size'],
                     #'subsampling_scheme': [0.1,0.9],
                     #'frobenius_lipschitz': [True],
                     'random_state':[random_state]}
        
    #Leave code below as is...
    param_grid['L21Reg'] = param_grid_L21Reg
    param_grid['L21GMMReg'] = param_grid_L21GMMReg
    param_grid['L21DGMMReg'] = param_grid_L21DGMMReg
    param_grid['Linear Regression'] = {}
    param_grid['Elastic Net'] = param_grid_EN
    param_grid['Group Lasso'] = param_grid_GL
    return param_grid

if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],int(sys.argv[7]),sys.argv[8],sys.argv[9],sys.argv[10],int(sys.argv[11]),int(sys.argv[12]),sys.argv[13],sys.argv[14],int(sys.argv[15]),int(sys.argv[16]),int(sys.argv[17]),int(sys.argv[18]),sys.argv[19],sys.argv[20],sys.argv[21],sys.argv[22])
