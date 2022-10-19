"""
File: StatsJob.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 6 of STREAMLINE - This 'Job' script is called by StatsMain.py and creates summaries of ML classification evaluation statistics
            (means and standard deviations), ROC and PRC plots (comparing CV performance in the same ML algorithm and comparing average performance
            between ML algorithms), model feature importance averages over CV runs, boxplots comparing ML algorithms for each metric, Kruskal Wallis
            and Mann Whitney statistical comparsions between ML algorithms, model feature importance boxplots for each algorithm, and composite feature
            importance plots summarizing model feature importance across all ML algorithms. It is run for a single dataset from the original target
            dataset folder (data_path) in Phase 1 (i.e. stats summary completed for all cv datasets).
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import sys
import time
import pandas as pd
import glob
import numpy as np
from scipy import stats
#from scipy import interp,stats
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from sklearn.metrics import auc
import csv
from statistics import mean,stdev
import pickle
import copy
from sklearn import tree
from subprocess import call
import scipy.stats

# from IPython.display import display_pdf
    
def job_reg(full_path,plot_FI_box,class_label,instance_label,cv_partitions,scale_data,plot_metric_boxplots,primary_metric,top_model_features,sig_cutoff,metric_weight,jupyterRun):
    """ Run all elements of stats summary and analysis for one one the original phase 1 datasets: summaries of average and standard deviations for all metrics and modeling algorithms,
    ROC and PRC plots (comparing CV performance in the same ML algorithm and comparing average performance between ML algorithms), model feature importance averages over CV runs,
    boxplots comparing ML algorithms for each metric, Kruskal Wallis and Mann Whitney statistical comparsions between ML algorithms, model feature importance boxplots for each
    algorithm, and composite feature importance plots summarizing model feature importance across all ML algorithms"""
    job_start_time = time.time() #for tracking phase runtime
    data_name = full_path.split('/')[-1]
    experiment_path = '/'.join(full_path.split('/')[:-1])
    if eval(jupyterRun):
        print('Running Statistics Summary for '+str(data_name))
    #Unpickle algorithm information from previous phase
    file = open(experiment_path+'/'+"algInfo.pickle", 'rb')
    algInfo = pickle.load(file)
    file.close()
    #Translate metric name from scikitlearn standard (currently balanced accuracy is hardcoded for use in generating FI plots due to no-skill normalization)
    #metric_term_dict = {'balanced_accuracy': 'Balanced Accuracy','accuracy': 'Accuracy','f1': 'F1_Score','recall': 'Sensitivity (Recall)','precision': 'Precision (PPV)','roc_auc': 'ROC_AUC'}
    #primary_metric = metric_term_dict[primary_metric] #currently not used
    metric_term_dict = {'max_error': 'Max Error','mean_absolute_error': 'Mean Absolute Error','mean_squared_error':'Mean Squared Error', 'median_absolute_error':'Median Absolute Error', 'explained_variance':'Explained Variance', 'pearson_correlation':'Pearson Correlation', 'f1': 'F1_Score'}
    metric_weight = metric_term_dict[metric_weight] #currently not used
    
    #Get algorithms run, specify algorithm abbreviations, colors to use for algorithms in plots, and original ordered feature name list
    algorithms,abbrev,colors,original_headers = preparation(full_path,algInfo)
    
    #Gather and summarize all evaluation metrics for each algorithm across all CVs. Returns result_table used to plot average ROC and PRC plots and metric_dict organizing all metrics over all algorithms and CVs.
    #result_table,metric_dict = primaryStats(algorithms,original_headers,cv_partitions,full_path,data_name,instance_label,class_label,abbrev,colors,plot_ROC,plot_PRC,jupyterRun)
    result_table,metric_dict = primaryStatsReg(algorithms,original_headers,cv_partitions,full_path,data_name,instance_label,class_label,abbrev,colors,jupyterRun)
    residualReg(algorithms,original_headers,cv_partitions,full_path,data_name,instance_label,class_label,abbrev,colors,jupyterRun)
    # result_table_2,metric_dict_2 = primaryStats_2(algorithms_2,original_headers,cv_partitions,full_path,data_name,instance_label,class_label,abbrev_2,colors_2,plot_ROC,plot_PRC,jupyterRun)
    #Plot ROC and PRC curves comparing average ML algorithm performance (averaged over all CVs)
    # if eval(jupyterRun):
    #     print('Generating ROC and PRC plots...')
    # doPlotROC(result_table,colors,full_path,jupyterRun)
    # doPlotPRC(result_table,colors,full_path,data_name,instance_label,class_label,jupyterRun)
    #Make list of metric names
    if eval(jupyterRun):
        print('Saving Metric Summaries...')
    metrics = list(metric_dict[algorithms[0]].keys())
    # metrics_2 = list(metric_dict_2[algorithms_2[0]].keys())
    #Save metric means and standard deviations
    # saveCorrelation(full_path,metrics_2,metric_dict_2)
    saveMetricMeans(full_path,metrics,metric_dict)
    saveMetricStd(full_path,metrics,metric_dict)
    #Generate boxplots comparing algorithm performance for each standard metric, if specified by user
    if eval(plot_metric_boxplots):
        if eval(jupyterRun):
            print('Generating Metric Boxplots...')
        metricBoxplots(full_path,metrics,algorithms,metric_dict,jupyterRun)
    #Calculate and export Kruskal Wallis, Mann Whitney, and wilcoxon Rank sum stats if more than one ML algorithm has been run (for the comparison) - note stats are based on comparing the multiple CV models for each algorithm.
    if len(algorithms) > 1:
        if eval(jupyterRun):
            print('Running Non-Parametric Statistical Significance Analysis...')
        kruskal_summary = kruskalWallis(full_path,metrics,algorithms,metric_dict,sig_cutoff)
        wilcoxonRank(full_path,metrics,algorithms,metric_dict,kruskal_summary,sig_cutoff)
        mannWhitneyU(full_path,metrics,algorithms,metric_dict,kruskal_summary,sig_cutoff)
    #Prepare for feature importance visualizations
    if eval(jupyterRun):
        print('Preparing for Model Feature Importance Plotting...')
    fi_df_list,fi_ave_list,fi_ave_norm_list,ave_metric_list,all_feature_list,non_zero_union_features,non_zero_union_indexes = prepFI(algorithms,full_path,abbrev,metric_dict,metric_weight) #old - 'Balanced Accuracy'
    #Select 'top' features for composite vizualization
    featuresToViz = selectForCompositeViz(top_model_features,non_zero_union_features,non_zero_union_indexes,algorithms,ave_metric_list,fi_ave_norm_list)
    #Generate FI boxplots for each modeling algorithm if specified by user
    if eval(plot_FI_box):
        if eval(jupyterRun):
            print('Generating Feature Importance Boxplots and Histograms...')
        doFIBoxplots(full_path,fi_df_list,fi_ave_list,algorithms,original_headers,top_model_features,jupyterRun)
        doFI_Histogram(full_path, fi_ave_list, algorithms, jupyterRun)
    #Visualize composite FI - Currently set up to only use Balanced Accuracy for composite FI plot visualization
    if eval(jupyterRun):
        print('Generating Composite Feature Importance Plots...')
    #Take top feature names to vizualize and get associated feature importance values for each algorithm, and original data ordered feature names list
    top_fi_ave_norm_list,all_feature_listToViz = getFI_To_Viz_Sorted(featuresToViz,all_feature_list,algorithms,fi_ave_norm_list) #If we want composite FI plots to be displayed in descenting total bar height order.
    #Generate Normalized composite FI plot
    composite_FI_plot(top_fi_ave_norm_list, algorithms, list(colors.values()), all_feature_listToViz, 'Norm',full_path,jupyterRun, 'Normalized Feature Importance')
    #Fractionate FI scores for normalized and fractionated composite FI plot
    fracLists = fracFI(top_fi_ave_norm_list)
    #Generate Normalized and Fractioned composite FI plot
    composite_FI_plot(fracLists, algorithms, list(colors.values()), all_feature_listToViz, 'Norm_Frac',full_path,jupyterRun, 'Normalized and Fractioned Feature Importance')
    #Weight FI scores for normalized and (model performance) weighted composite FI plot
    weightedLists,weights = weightFI_2(ave_metric_list,top_fi_ave_norm_list)
    #Generate Normalized and Weighted Compount FI plot
    composite_FI_plot(weightedLists, algorithms, list(colors.values()), all_feature_listToViz, 'Norm_Weight',full_path,jupyterRun, 'Normalized and Weighted Feature Importance')
    #Weight the Fractionated FI scores for normalized,fractionated, and weighted compount FI plot
    weightedFracLists = weightFracFI(fracLists,weights)
    #Generate Normalized, Fractionated, and Weighted Compount FI plot
    composite_FI_plot(weightedFracLists, algorithms, list(colors.values()), all_feature_listToViz, 'Norm_Frac_Weight',full_path,jupyterRun, 'Normalized, Fractioned, and Weighted Feature Importance')
    
    
    
    #Export phase runtime
    saveRuntime(full_path,job_start_time)
    #Parse all pipeline runtime files into a single runtime report
    parseRuntime(full_path,algorithms,abbrev)
    # Print phase completion
    print(data_name + " phase 5 complete")
    job_file = open(experiment_path + '/jobsCompleted/job_stats_' + data_name + '.txt', 'w')
    job_file.write('complete')
    job_file.close()
    
def preparation(full_path,algInfo):
    """ Creates directory for all results files, decodes included ML modeling algorithms that were run, specifies figure abbreviations for algorithms
    and color to use for each algorithm in plots, and loads original ordered feature name list to use as a reference to facilitate combining feature
    importance results across cv runs where different features may have been dropped during the feature selection phase."""
    #Create Directory
    if not os.path.exists(full_path+'/model_evaluation'):
        os.mkdir(full_path+'/model_evaluation')

    #Extract the original algorithm name, abreviated name, and color to use for each algortithm run by users
    algorithms = []
    abbrev = {}
    colors = {}
    for key in algInfo:
        if algInfo[key][0]: # If that algorithm was used
        #   if key != 'L21Reg' and key !='L21GMMReg' and key !='L21DGMMReg':
              algorithms.append(key)
              abbrev[key] = (algInfo[key][1])
              colors[key] = (algInfo[key][2])
    print(algorithms)
    original_headers = pd.read_csv(full_path+"/exploratory/OriginalFeatureNames.csv",sep=',').columns.values.tolist() #Get Original Headers
    return algorithms,abbrev,colors,original_headers

def residualReg(algorithms,original_headers,cv_partitions,full_path,data_name,instance_label,class_label,abbrev,colors,jupyterRun):
    s_res_trains = []
    s_res_tests = []
    s_y_train_preds = []
    s_y_test_preds = []
    s_y_trains = []
    s_y_tests = []
    
    m_trains = []
    b_trains = []
    m_tests = []
    b_tests = []
    for algorithm in algorithms:
        s_res_train = []
        s_res_test = []
        s_y_train_pred = []
        s_y_test_pred = []
        s_y_train = []
        s_y_test = []
        for cvCount in range(0,cv_partitions):
            result_file = full_path+'/model_evaluation/pickled_metrics/'+abbrev[algorithm]+"_CV_"+str(cvCount)+"_residuals.pickle"
            file = open(result_file, 'rb')
            results = pickle.load(file)
            file.close()
            res_train = results[0]
            res_test = results[1]
            y_train_pred = results[2]
            y_test_pred = results[3]
            y_train = results[4]
            y_test = results[5]
            
            s_res_train = np.stack([res_train], axis = 0)
            s_res_test = np.stack([res_test], axis = 0)
            s_y_train_pred = np.stack([y_train_pred], axis = 0)
            s_y_test_pred = np.stack([y_test_pred], axis = 0)
            s_y_train = np.stack([y_train], axis = 0)
            s_y_test = np.stack([y_test], axis = 0)
            
        s_res_train = s_res_train[0]
        s_res_test = s_res_test[0]
        s_y_train_pred = s_y_train_pred[0]
        s_y_test_pred = s_y_test_pred[0]
        s_y_train = s_y_train[0]
        s_y_test = s_y_test[0]
        
        s_res_trains.append(s_res_train)
        s_res_tests.append(s_res_test)
        s_y_train_preds.append(y_train_pred)
        s_y_test_preds.append(y_test_pred)
        s_y_trains.append(y_train)
        s_y_tests.append(s_y_test)
        
        n_bins = 20
        plt.figure()
        plt.rcParams.update({'font.size': 15})
        fig, axes = plt.subplots(1, 2, sharey = True, figsize = [10, 5])
        axes[0].scatter(s_y_train_pred, s_res_train, alpha = 0.5, c = 'cornflowerblue', label = 'Training')
        axes[0].scatter(s_y_test_pred, s_res_test, alpha = 0.5, c = 'firebrick', label = 'Testing')
        axes[0].axhline(y = 0, color = 'black', linestyle = '-')
        axes[1].hist(s_res_train, bins = n_bins, alpha = 0.5, color = 'cornflowerblue', orientation = 'horizontal')
        axes[1].hist(s_res_test, bins = n_bins, alpha = 0.5, color = 'firebrick', orientation = 'horizontal')
        axes[1].axhline(y = 0, color = 'black', linestyle = '-')
        axes[0].set_ylabel('Residual')
        axes[0].set_xlabel('Predicted Cognition Score')
        axes[1].set_xlabel('Number of occurrence')
        fig.suptitle('Distribution of Residual (' + algorithm+')')
        fig.legend(loc = 'upper right')
        plt.show()
        if not os.path.exists(full_path + '/model_evaluation/residualPlot'):
            os.mkdir(full_path + '/model_evaluation/residualPlot')
        fig.savefig(full_path + '/model_evaluation/residualPlot/'+algorithm+'_residual_distrib.png')
        
        
        m_1, b_1 = np.polyfit(s_y_train_pred, s_y_train, 1)
        m_2, b_2 = np.polyfit(s_y_test_pred, s_y_test, 1)
        m_trains.append(m_1)
        m_tests.append(m_2)
        b_trains.append(b_1)
        b_tests.append(b_2)
        
        plt.figure()
        plt.rcParams.update({'font.size': 15})
        fig_1, axes_1 = plt.subplots(1, 2, sharey = True, figsize = [10, 5])
        axes_1[0].scatter(s_y_train_pred, s_y_train, alpha = 0.5, c = 'cornflowerblue', label = 'Training')
        axes_1[1].scatter(s_y_test_pred, s_y_test, alpha = 0.5, c = 'firebrick', label = 'Testing')
        axes_1[0].plot(s_y_train_pred, m_1*s_y_train_pred+b_1)
        axes_1[1].plot(s_y_test_pred, m_2*s_y_test_pred+b_2)
        axes_1[0].set_ylabel('Actual Cognition Score')
        axes_1[0].set_xlabel('Predicted Cognition Score')
        axes_1[1].set_xlabel('Predicted Cognition Score')
        fig_1.suptitle('Actual Score vs. Predicted Score (' + algorithm+')')
        fig_1.legend(loc = 'upper right')
        plt.show()
        if not os.path.exists(full_path + '/model_evaluation/actualPredictPolt'):
            os.mkdir(full_path + '/model_evaluation/actualPredictPolt')
        fig_1.savefig(full_path + '/model_evaluation/actualPredictPolt/'+algorithm+'_actual_vs_predict.png')
        
        
    plt.figure()
    fig_2, axes_2 = plt.subplots(2, 2, sharey = True, figsize = [20, 15])
    n_bins = 20
    for i in range(len(algorithms)):
        axes_2[0, 0].scatter(s_y_train_preds[i], s_res_trains[i], alpha = 0.4, c = colors[algorithms[i]])
        axes_2[0, 1].hist(s_res_trains[i], bins = n_bins, alpha = 0.7, color = colors[algorithms[i]], orientation = 'horizontal', label = algorithms[i])
        axes_2[1, 0].scatter(s_y_test_preds[i], s_res_tests[i], alpha = 0.4, c = colors[algorithms[i]])
        axes_2[1, 1].hist(s_res_tests[i], bins = n_bins, alpha = 0.7, color = colors[algorithms[i]], orientation = 'horizontal')
    axes_2[0, 0].axhline(y = 0, color = 'black', linestyle = '-')
    axes_2[0, 1].axhline(y = 0, color = 'black', linestyle = '-')
    axes_2[1, 0].axhline(y = 0, color = 'black', linestyle = '-')
    axes_2[1, 1].axhline(y = 0, color = 'black', linestyle = '-')
    axes_2[0, 0].title.set_text("Residual vs Predicted Cognition Score (Training)")
    axes_2[1, 0].title.set_text("Residual vs Predicted Cognition Score (Testing)")
    axes_2[0, 1].title.set_text("Residual Distribution (Training)")
    axes_2[1, 1].title.set_text("Residual Distribution (Testing)")
    axes_2[0, 0].set_ylabel('Residual')
    axes_2[1, 0].set_ylabel('Residual')
    axes_2[1, 0].set_xlabel('Predicted Cognition Score')
    axes_2[1, 1].set_xlabel('Predicted Cognition Score')
    fig_2.legend(loc = 'upper right')
    plt.show()
    fig_2.savefig(full_path + '/model_evaluation/residualPlot/residual_distrib_all_algorithms.png')
    
    fig_3, axes_3 = plt.subplots(1, 2, sharey = True, figsize = [20, 10])
    for i in range(len(algorithms)):
        axes_3[0].scatter(s_y_train_preds[i], s_y_trains[i], alpha = 0.3, c = colors[algorithms[i]])
        axes_3[1].scatter(s_y_test_preds[i], s_y_tests[i], alpha = 0.3, c = colors[algorithms[i]])
        axes_3[0].plot(s_y_train_preds[i], m_trains[i] * s_y_train_preds[i] + b_trains[i], color = colors[algorithms[i]], label = algorithms[i])
        axes_3[1].plot(s_y_test_preds[i], m_tests[i] * s_y_test_preds[i] + b_tests[i], color = colors[algorithms[i]])
    axes_3[0].title.set_text('Actual Score vs. Predicted Score (Train)')
    axes_3[1].title.set_text('Actual Score vs. Predicted Score (Test)')
    axes_3[0].set_ylabel('Actual Cognition Score')
    axes_3[0].set_xlabel('Predicted Cognition Score')
    axes_3[1].set_xlabel('Predicted Cognition Score')
    fig_3.legend(loc = 'upper right')
    plt.show()
    fig_3.savefig(full_path + '/model_evaluation/residualPlot/actual_vs_predict_all_algorithms.png')
    
    fig_4, axes_4 = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(len(algorithms)):
        stats.probplot(s_res_trains[i], dist=stats.norm, sparams=(2,3), plot=plt, fit=False)
    for i in range(len(algorithms)):
        # axes_4.get_lines()[i].remove()
        axes_4.get_lines()[i].set_markerfacecolor(colors[algorithms[i]])
        axes_4.get_lines()[i].set_alpha(0.5)
        axes_4.get_lines()[i].set_color(colors[algorithms[i]])
        axes_4.get_lines()[i].set_label(algorithms[i])
    axes_4.title.set_text("Probability Plot of Training Residual")
    axes_4.set_xlabel("Theoretical Quantiles")
    axes_4.set_ylabel("Ordered Residual")
    axes_4.legend(loc = 'upper right')
    plt.show()
    fig_4.savefig(full_path + '/model_evaluation/residualPlot/probability_train_residual_all_algorithms.png')
    
    fig_5, axes_5 = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(len(algorithms)):
        stats.probplot(s_res_tests[i], dist=stats.norm, sparams=(2,3), plot=plt, fit=False)
    for i in range(len(algorithms)):
        # axes_4.get_lines()[i].remove()
        axes_5.get_lines()[i].set_markerfacecolor(colors[algorithms[i]])
        axes_5.get_lines()[i].set_alpha(0.5)
        axes_5.get_lines()[i].set_color(colors[algorithms[i]])
        axes_5.get_lines()[i].set_label(algorithms[i])
    axes_5.title.set_text("Probability Plot of Training Residual")
    axes_5.set_xlabel("Theoretical Quantiles")
    axes_5.set_ylabel("Ordered Residual")
    axes_5.legend(loc = 'upper right')
    plt.show()
    fig_5.savefig(full_path + '/model_evaluation/residualPlot/probability_test_residual_all_algorithms.png')
    
    
def primaryStatsReg(algorithms,original_headers,cv_partitions,full_path,data_name,instance_label,class_label,abbrev,colors,jupyterRun):
    """ Combine classification metrics and model feature importance scores as well as ROC and PRC plot data across all CV datasets.
    Generate ROC and PRC plots comparing separate CV models for each individual modeling algorithm."""
    result_table = []
    metric_dict = {}
    for algorithm in algorithms: #completed for each individual ML modeling algorithm
        # Define evaluation stats variable lists
        s_me = [] # balanced accuracies
        s_mae = [] # standard accuracies
        s_mse = [] # F1 scores
        s_mdae = [] # recall values
        s_mape = [] # specificities
        s_evs = [] # precision values
        s_corr = [] # true positives
        # Define feature importance lists
        FI_all = [] # used to save model feature importances individually for each cv within single summary file (all original features in dataset prior to feature selection included)
        mes = []
        maes = []
        mses = []
        mdaes = []
        mapes = []
        evss = []
        corrs = []
        #Gather statistics over all CV partitions
        for cvCount in range(0,cv_partitions):
            #Unpickle saved metrics from previous phase
            result_file = full_path+'/model_evaluation/pickled_metrics/'+abbrev[algorithm]+"_CV_"+str(cvCount)+"_metrics.pickle"
            file = open(result_file, 'rb')
            results = pickle.load(file)
            file.close()
            #Separate pickled results
            me = results[0][0]
            mae = results[0][1]
            mse = results[0][2]
            mdae = results[0][3]
            evs = results[0][4]
            corr = results[0][5]
            fi = results[1]


            mes.append(me)
            maes.append(mae)
            mses.append(mse)
            mdaes.append(mdae)
            evss.append(evs)
            corrs.append(corr)
            # Format feature importance scores as list (takes into account that all features are not in each CV partition)
            tempList = []
            j = 0
            headers = pd.read_csv(full_path+'/CVDatasets/'+data_name+'_CV_'+str(cvCount)+'_Test.csv').columns.values.tolist()
            if instance_label != 'None':
                headers.remove(instance_label)
            headers.remove(class_label)
            for each in original_headers:
                if each in headers:  # Check if current feature from original dataset was in the partition
                    # Deal with features not being in original order (find index of current feature list.index()
                    f_index = headers.index(each)
                    tempList.append(fi[f_index])
                else:
                    tempList.append(0)
                j += 1
            FI_all.append(tempList)

        if jupyterRun:
            print(algorithm)
        
        mean_me = np.mean(mes, axis=0)
        mean_mae = np.mean(maes, axis=0)
        mean_mse = np.mean(mses, axis=0)
        mean_mdae = np.mean(mdaes, axis = 0)
        mean_evs = np.mean(evss, axis=0)
        mean_corr = np.mean(corrs, axis=0)

        #Export and save all CV metric stats for each individual algorithm  -----------------------------------------------------------------------------
        results = {'Max Error': mes, 'Mean Absolute Error': maes, 'Mean Squared Error': mses, 'Median Absolute Error': mdaes, 'Explained Variance': evss, 'Pearson Correlation': corrs}
        dr = pd.DataFrame(results)
        filepath = full_path+'/model_evaluation/'+abbrev[algorithm]+"_performance.csv"
        dr.to_csv(filepath, header=True, index=False)
        metric_dict[algorithm] = results
        #Save Average FI Stats
        save_FI(FI_all, abbrev[algorithm], original_headers, full_path)

        result_dict = {'algorithm':algorithm,'max_error':mean_me, 'mean_absolute_error':mean_mae, 'mean_squared_error':mean_mse, 'median_absolute_error':mean_mdae, 'explained_variance':mean_evs, 'pearson_correlation': mean_corr}
        result_table.append(result_dict)
    #Result table later used to create global ROC an PRC plots comparing average ML algorithm performance.
    result_table = pd.DataFrame.from_dict(result_table)
    result_table.set_index('algorithm',inplace=True)
    return result_table,metric_dict
    

def save_FI(FI_all,algorithm,globalFeatureList,full_path):
    """ Creates directory to store model feature importance results and, for each algorithm, exports a file of feature importance scores from each CV. """
    dr = pd.DataFrame(FI_all)
    if not os.path.exists(full_path+'/model_evaluation/feature_importance/'):
        os.mkdir(full_path+'/model_evaluation/feature_importance/')
    filepath = full_path+'/model_evaluation/feature_importance/'+algorithm+"_FI.csv"
    dr.to_csv(filepath, header=globalFeatureList, index=False)

def doPlotROC(result_table,colors,full_path,jupyterRun):
    """ Generate ROC plot comparing average ML algorithm performance (over all CV training/testing sets)"""
    count = 0
    #Plot curves for each individual ML algorithm
    for i in result_table.index:
        #plt.plot(result_table.loc[i]['fpr'],result_table.loc[i]['tpr'], color=colors[i],label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
        plt.plot(result_table.loc[i]['fpr'],result_table.loc[i]['tpr'], color=colors[i],label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
        count += 1
    # Set figure dimensions
    plt.rcParams["figure.figsize"] = (6,6)
    # Plot no-skill line
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='No-Skill', alpha=.8)
    #Specify plot axes,labels, and legend
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01,1))
    #Export and/or show plot
    plt.savefig(full_path+'/model_evaluation/Summary_ROC.png', bbox_inches="tight")
    if eval(jupyterRun):
        plt.show()
    else:
        plt.close('all')

def doPlotPRC(result_table,colors,full_path,data_name,instance_label,class_label,jupyterRun):
    """ Generate PRC plot comparing average ML algorithm performance (over all CV training/testing sets)"""
    count = 0
    #Plot curves for each individual ML algorithm
    for i in result_table.index:
        plt.plot(result_table.loc[i]['recall'],result_table.loc[i]['prec'], color=colors[i],label="{}, AUC={:.3f}, APS={:.3f}".format(i, result_table.loc[i]['pr_auc'],result_table.loc[i]['ave_prec']))
        count += 1
    #Estimate no skill line based on the fraction of cases found in the first test dataset
    test = pd.read_csv(full_path+'/CVDatasets/'+data_name+'_CV_0_Test.csv')
    if instance_label != 'None':
        test = test.drop(instance_label, axis=1)
    testY = test[class_label].values
    noskill = len(testY[testY == 1]) / len(testY)  # Fraction of cases
    # Plot no-skill line
    plt.plot([0, 1], [noskill, noskill], color='black', linestyle='--',label='No-Skill', alpha=.8)
    #Specify plot axes,labels, and legend
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Recall (Sensitivity)", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("Precision (PPV)", fontsize=15)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01,1))
    #Export and/or show plot
    plt.savefig(full_path+'/model_evaluation/Summary_PRC.png', bbox_inches="tight")
    if eval(jupyterRun):
        plt.show()
    else:
        plt.close('all')

def saveCorrelation(full_path,metrics,metric_dict):
    with open(full_path+'/model_evaluation/Summary_performance_mean.csv', mode = 'w', newline="") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        e = ['']
        e.extend(metrics)
        writer.writerow(e)
        for algorithm in metric_dict:
            astats = []
            for l in list(metric_dict[algorithm].values()):
                l = [float(i) for i in l]
                meani = round(mean(l), 3)
                astats.append(str(meani))
            toAdd = [algorithm]
            toAdd.extend(astats)
            writer.writerow(toAdd)
    file.close()
    
def saveMetricMeans(full_path,metrics,metric_dict):
    """ Exports csv file with average metric values (over all CVs) for each ML modeling algorithm"""
    with open(full_path+'/model_evaluation/Summary_performance_mean.csv',mode='w', newline="") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        e = ['']
        e.extend(metrics)
        writer.writerow(e) #Write headers (balanced accuracy, etc.)
        for algorithm in metric_dict:
            astats = []
            for l in list(metric_dict[algorithm].values()):
                l = [float(i) for i in l]
                meani = mean(l)
                std = stdev(l)
                astats.append(str(meani))
            toAdd = [algorithm]
            toAdd.extend(astats)
            writer.writerow(toAdd)
    file.close()

def saveMetricStd(full_path,metrics,metric_dict):
    """ Exports csv file with metric value standard deviations (over all CVs) for each ML modeling algorithm"""
    with open(full_path + '/model_evaluation/Summary_performance_std.csv', mode='w', newline="") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        e = ['']
        e.extend(metrics)
        writer.writerow(e)  # Write headers (balanced accuracy, etc.)
        for algorithm in metric_dict:
            astats = []
            for l in list(metric_dict[algorithm].values()):
                l = [float(i) for i in l]
                std = stdev(l)
                astats.append(str(std))
            toAdd = [algorithm]
            toAdd.extend(astats)
            writer.writerow(toAdd)
    file.close()

def metricBoxplots(full_path,metrics,algorithms,metric_dict,jupyterRun):
    """ Export boxplots comparing algorithm performance for each standard metric"""
    if not os.path.exists(full_path + '/model_evaluation/metricBoxplots'):
        os.mkdir(full_path + '/model_evaluation/metricBoxplots')
    for metric in metrics:
        tempList = []
        for algorithm in algorithms:
            tempList.append(metric_dict[algorithm][metric])
        td = pd.DataFrame(tempList)
        td = td.transpose()
        td.columns = algorithms
        #Generate boxplot
        boxplot = td.boxplot(column=algorithms,rot=90)
        #Specify plot labels
        plt.ylabel(str(metric))
        plt.xlabel('ML Algorithm')
        #Export and/or show plot
        plt.savefig(full_path + '/model_evaluation/metricBoxplots/Compare_'+metric+'.png', bbox_inches="tight")
        if eval(jupyterRun):
            plt.show()
        else:
            plt.close('all')

def kruskalWallis(full_path,metrics,algorithms,metric_dict,sig_cutoff):
    """ Apply non-parametric Kruskal Wallis one-way ANOVA on ranks. Determines if there is a statistically significant difference in algorithm performance across CV runs.
    Completed for each standard metric separately."""
    # Create directory to store significance testing results (used for both Kruskal Wallis and MannWhitney U-test)
    if not os.path.exists(full_path + '/model_evaluation/statistical_comparisons'):
        os.mkdir(full_path + '/model_evaluation/statistical_comparisons')
    #Create dataframe to store analysis results for each metric
    label = ['Statistic', 'P-Value', 'Sig(*)']
    kruskal_summary = pd.DataFrame(index=metrics, columns=label)
    #Apply Kruskal Wallis test for each metric
    for metric in metrics:
        tempArray = []
        for algorithm in algorithms:
            tempArray.append(metric_dict[algorithm][metric])
        try:
            result = stats.kruskal(*tempArray)
        except:
            result = [tempArray[0],1]
        kruskal_summary.at[metric, 'Statistic'] = str(round(result[0], 6))
        kruskal_summary.at[metric, 'P-Value'] = str(round(result[1], 6))
        if result[1] < sig_cutoff:
            kruskal_summary.at[metric, 'Sig(*)'] = str('*')
        else:
            kruskal_summary.at[metric, 'Sig(*)'] = str('')
    #Export analysis summary to .csv file
    kruskal_summary.to_csv(full_path + '/model_evaluation/statistical_comparisons/KruskalWallis.csv')
    return kruskal_summary

def wilcoxonRank(full_path,metrics,algorithms,metric_dict,kruskal_summary,sig_cutoff):
    """ Apply non-parametric Wilcoxon signed-rank test (pairwise comparisons). If a significant Kruskal Wallis algorithm difference was found for a given metric, Wilcoxon tests individual algorithm pairs
    to determine if there is a statistically significant difference in algorithm performance across CV runs. Test statistic will be zero if all scores from one set are
    larger than the other."""
    for metric in metrics:
        if kruskal_summary['Sig(*)'][metric] == '*':
            wilcoxon_stats = []
            done = []
            for algorithm1 in algorithms:
                for algorithm2 in algorithms:
                    if not [algorithm1,algorithm2] in done and not [algorithm2,algorithm1] in done and algorithm1 != algorithm2:
                        set1 = metric_dict[algorithm1][metric]
                        set2 = metric_dict[algorithm2][metric]
                        #handle error when metric values are equal for both algorithms
                        if set1 == set2:  # Check if all nums are equal in sets
                            report = ['NA',1]
                        else: # Apply Wilcoxon Rank Sum test
                            report = stats.wilcoxon(set1,set2)
                        #Summarize test information in list
                        tempstats = [algorithm1,algorithm2,report[0],report[1],'']
                        if report[1] < sig_cutoff:
                            tempstats[4] = '*'
                        wilcoxon_stats.append(tempstats)
                        done.append([algorithm1,algorithm2])
            #Export test results
            wilcoxon_stats_df = pd.DataFrame(wilcoxon_stats)
            wilcoxon_stats_df.columns = ['Algorithm 1', 'Algorithm 2', 'Statistic', 'P-Value', 'Sig(*)']
            wilcoxon_stats_df.to_csv(full_path + '/model_evaluation/statistical_comparisons/WilcoxonRank_'+metric+'.csv', index=False)

def mannWhitneyU(full_path,metrics,algorithms,metric_dict,kruskal_summary,sig_cutoff):
    """ Apply non-parametric Mann Whitney U-test (pairwise comparisons). If a significant Kruskal Wallis algorithm difference was found for a given metric, Mann Whitney tests individual algorithm pairs
    to determine if there is a statistically significant difference in algorithm performance across CV runs. Test statistic will be zero if all scores from one set are
    larger than the other."""
    for metric in metrics:
        if kruskal_summary['Sig(*)'][metric] == '*':
            mann_stats = []
            done = []
            for algorithm1 in algorithms:
                for algorithm2 in algorithms:
                    if not [algorithm1,algorithm2] in done and not [algorithm2,algorithm1] in done and algorithm1 != algorithm2:
                        set1 = metric_dict[algorithm1][metric]
                        set2 = metric_dict[algorithm2][metric]
                        if set1 == set2:  # Check if all nums are equal in sets
                            report = ['NA',1]
                        else: #Apply Mann Whitney U test
                            report = stats.mannwhitneyu(set1,set2)
                        #Summarize test information in list
                        tempstats = [algorithm1,algorithm2,report[0],report[1],'']
                        if report[1] < sig_cutoff:
                            tempstats[4] = '*'
                        mann_stats.append(tempstats)
                        done.append([algorithm1,algorithm2])
            #Export test results
            mann_stats_df = pd.DataFrame(mann_stats)
            mann_stats_df.columns = ['Algorithm 1', 'Algorithm 2', 'Statistic', 'P-Value', 'Sig(*)']
            mann_stats_df.to_csv(full_path + '/model_evaluation/statistical_comparisons/MannWhitneyU_'+metric+'.csv', index=False)


def prepFI(algorithms,full_path,abbrev,metric_dict,metric_weight):
    """ Organizes and prepares model feature importance data for boxplot and composite feature importance figure generation."""
    #Initialize required lists
    fi_df_list = []         # algorithm feature importance dataframe list (used to generate FI boxplots for each algorithm)
    fi_ave_list = []        # algorithm feature importance averages list (used to generate composite FI barplots)
    ave_metric_list = []    # algorithm focus metric averages list (used in weighted FI viz)
    all_feature_list = []   # list of pre-feature selection feature names as they appear in FI reports for each algorithm
    #Get necessary feature importance data and primary metric data (currenly only 'balanced accuracy' can be used for this)
    for algorithm in algorithms:
        # Get relevant feature importance info
        temp_df = pd.read_csv(full_path+'/model_evaluation/feature_importance/'+abbrev[algorithm]+"_FI.csv") #CV FI scores for all original features in dataset.
        if algorithm == algorithms[0]:  # Should be same for all algorithm files (i.e. all original features in standard CV dataset order)
            all_feature_list = temp_df.columns.tolist()
        fi_df_list.append(temp_df)
        fi_ave_list.append(temp_df.mean().tolist()) #Saves average FI scores over CV runs
        # Get relevant metric info
        avgWeight = mean(metric_dict[algorithm][metric_weight]) #   old-     avgBA = mean(metric_dict[algorithm][primary_metric])
        ave_metric_list.append(avgWeight)
    #Normalize Average Feature importance scores so they fall between (0 - 1)
    fi_ave_norm_list = []
    for each in fi_ave_list:  # each algorithm
        normList = []
        for i in range(len(each)): #each feature (score) in original data order
            if each[i] <= 0: #Feature importance scores assumed to be uninformative if at or below 0
                normList.append(0)
            else:
                normList.append((each[i]) / (max(each)))
        fi_ave_norm_list.append(normList)
    #Identify features with non-zero averages (step towards excluding features that had zero feature importance for all algorithms)
    alg_non_zero_FI_list = [] #stores list of feature name lists that are non-zero for each algorithm
    for each in fi_ave_list:  # each algorithm
        temp_non_zero_list = []
        for i in range(len(each)):  # each feature
            if each[i] > 0.0:
                temp_non_zero_list.append(all_feature_list[i]) #add feature names with positive values (doesn't need to be normalized for this)
        alg_non_zero_FI_list.append(temp_non_zero_list)
    non_zero_union_features = alg_non_zero_FI_list[0]  # grab first algorithm's list
    #Identify union of features with non-zero averages over all algorithms (i.e. if any algorithm found a non-zero score it will be considered for inclusion in top feature visualizations)
    for j in range(1, len(algorithms)):
        non_zero_union_features = list(set(non_zero_union_features) | set(alg_non_zero_FI_list[j]))
    non_zero_union_indexes = []
    for i in non_zero_union_features:
        non_zero_union_indexes.append(all_feature_list.index(i))
    return fi_df_list,fi_ave_list,fi_ave_norm_list,ave_metric_list,all_feature_list,non_zero_union_features,non_zero_union_indexes

def selectForCompositeViz(top_model_features,non_zero_union_features,non_zero_union_indexes,algorithms,ave_metric_list,fi_ave_norm_list):
    """ Identify list of top features over all algorithms to visualize (note that best features to vizualize are chosen using algorithm performance weighting and normalization:
    frac plays no useful role here only for viz). All features included if there are fewer than 'top_model_features'. Top features are determined by the sum of performance
    (i.e. balanced accuracy) weighted feature importances over all algorithms."""
    featuresToViz = None
    #Create performance weighted score sum dictionary for all features
    scoreSumDict = {}
    i = 0
    for each in non_zero_union_features:  # for each non-zero feature
        for j in range(len(algorithms)):  # for each algorithm
            # grab target score from each algorithm
            score = fi_ave_norm_list[j][non_zero_union_indexes[i]]
            # multiply score by algorithm performance weight
            weight = ave_metric_list[j]
            # if weight <= .5: #This is why this method is limited to balanced_accuracy and roc_auc
            #     weight = 0
            # if not weight == 0:
                # weight = (weight - 0.5) / 0.5
            score = score * weight
            #score = score * ave_metric_list[j]
            if not each in scoreSumDict:
                scoreSumDict[each] = score
            else:
                scoreSumDict[each] += score
        i += 1
    # Sort features by decreasing score
    scoreSumDict_features = sorted(scoreSumDict, key=lambda x: scoreSumDict[x], reverse=True)
    if len(non_zero_union_features) > top_model_features: #Keep all features if there are fewer than specified top results
        featuresToViz = scoreSumDict_features[0:top_model_features]
    else:
        featuresToViz = scoreSumDict_features
    return featuresToViz #list of feature names to vizualize in composite FI plots.

def doFIBoxplots(full_path,fi_df_list,fi_ave_list,algorithms,original_headers,top_model_features, jupyterRun):
    """ Generate individual feature importance boxplots for each algorithm """
    algorithmCounter = 0
    for algorithm in algorithms: #each algorithms
        #Make average feature importance score dicitonary
        scoreDict = {}
        counter = 0
        for ave_score in fi_ave_list[algorithmCounter]: #each feature
            scoreDict[original_headers[counter]] = ave_score
            counter += 1
        # Sort features by decreasing score
        scoreDict_features = sorted(scoreDict, key=lambda x: scoreDict[x], reverse=True)
        #Make list of feature names to vizualize
        if len(original_headers) > top_model_features:
            featuresToViz = scoreDict_features[0:top_model_features]
        else:
            featuresToViz = scoreDict_features
        # FI score dataframe for current algorithm
        df = fi_df_list[algorithmCounter]
        # Subset of dataframe (in ranked order) to vizualize
        viz_df = df[featuresToViz]
        #Generate Boxplot
        fig = plt.figure(figsize=(15, 4))
        boxplot = viz_df.boxplot(rot=90)
        plt.title(algorithm)
        plt.ylabel('Feature Importance Score')
        plt.xlabel('Features')
        plt.xticks(np.arange(1, len(featuresToViz) + 1), featuresToViz, rotation='vertical')
        plt.savefig(full_path+'/model_evaluation/feature_importance/' + algorithm + '_boxplot',bbox_inches="tight")
        if eval(jupyterRun):
            plt.show()
        else:
            plt.close('all')    #Identify and sort (decreaseing) features with top average FI
        algorithmCounter += 1

def doFI_Histogram(full_path, fi_ave_list, algorithms, jupyterRun):
    """ Generate histogram showing distribution of average feature importances scores for each algorithm. """
    algorithmCounter = 0
    for algorithm in algorithms: #each algorithms
        aveScores = fi_ave_list[algorithmCounter]
        #Plot a histogram of average feature importance
        plt.hist(aveScores,bins=100)
        plt.xlabel("Average Feature Importance")
        plt.ylabel("Frequency")
        plt.title("Histogram of Average Feature Importance for "+str(algorithm))
        plt.xticks(rotation = 'vertical')
        plt.savefig(full_path+'/model_evaluation/feature_importance/' + algorithm + '_histogram',bbox_inches="tight")
        if eval(jupyterRun):
            plt.show()
        else:
            plt.close('all')

def getFI_To_Viz_Sorted(featuresToViz,all_feature_list,algorithms,fi_ave_norm_list):
    """ Takes a list of top features names for vizualization, gets their indexes. In every composite FI plot features are ordered the same way
    they are selected for vizualization (i.e. normalized and performance weighted). Because of this feature bars are only perfectly ordered in
    descending order for the normalized + performance weighted composite plot. """
    #Get original feature indexs for selected feature names
    feature_indexToViz = [] #indexes of top features
    for i in featuresToViz:
        feature_indexToViz.append(all_feature_list.index(i))
    # Create list of top feature importance values in original dataset feature order
    top_fi_ave_norm_list = [] #feature importance values of top features for each algorithm (list of lists)
    for i in range(len(algorithms)):
        tempList = []
        for j in feature_indexToViz: #each top feature index
            tempList.append(fi_ave_norm_list[i][j]) #add corresponding FI value
        top_fi_ave_norm_list.append(tempList)
    all_feature_listToViz = featuresToViz
    return top_fi_ave_norm_list,all_feature_listToViz

def composite_FI_plot(fi_list, algorithms, algColors, all_feature_listToViz, figName,full_path,jupyterRun,yLabelText):
    """ Generate composite feature importance plot given list of feature names and associated feature importance scores for each algorithm.
    This is run for different transformations of the normalized feature importance scores. """
    # Set basic plot properites
    rc('font', weight='bold', size=16)
    # The position of the bars on the x-axis
    r = all_feature_listToViz #feature names
    #Set width of bars
    barWidth = 0.75
    #Set figure dimensions
    plt.figure(figsize=(24, 12))
    #Plot first algorithm FI scores (lowest) bar
    p1 = plt.bar(r, fi_list[0], color=algColors[0], edgecolor='white', width=barWidth)
    #Automatically calculate space needed to plot next bar on top of the one before it
    bottoms = [] #list of space used by previous algorithms for each feature (so next bar can be placed directly above it)
    for i in range(len(algorithms) - 1):
        for j in range(i + 1):
            if j == 0:
                bottom = np.array(fi_list[0])
            else:
                bottom += np.array(fi_list[j])
        bottoms.append(bottom)
    if not isinstance(bottoms, list):
        bottoms = bottoms.tolist()
    if len(algorithms) > 1:
        #Plot subsequent feature bars for each subsequent algorithm
        ps = [p1[0]]
        for i in range(len(algorithms) - 1):
            p = plt.bar(r, fi_list[i + 1], bottom=bottoms[i], color=algColors[i + 1], edgecolor='white', width=barWidth)
            ps.append(p[0])
        lines = tuple(ps)
    else:
        ps = [p1[0]]
        lines = tuple(ps)
    # Specify axes info and legend
    plt.xticks(np.arange(len(all_feature_listToViz)), all_feature_listToViz, rotation='vertical')
    plt.xlabel("Feature", fontsize=20)
    plt.ylabel(yLabelText, fontsize=20)
    #plt.legend(lines[::-1], algorithms[::-1],loc="upper left", bbox_to_anchor=(1.01,1)) #legend outside plot
    plt.legend(lines[::-1], algorithms[::-1],loc="upper right")
    #Export and/or show plot
    plt.savefig(full_path+'/model_evaluation/feature_importance/Compare_FI_' + figName + '.png', bbox_inches='tight')
    if eval(jupyterRun):
        plt.show()
    else:
        plt.close('all')

def fracFI(top_fi_ave_norm_list):
    """ Transforms feature scores so that they sum to 1 over all features for a given algorithm.  This way the normalized and fracionated composit bar plot
    offers equal total bar area for every algorithm. The intuition here is that if an algorithm gives the same FI scores for all top features it won't be
    overly represented in the resulting plot (i.e. all features can have the same maximum feature importance which might lead to the impression that an
    algorithm is working better than it is.) Instead, that maximum 'bar-real-estate' has to be divided by the total number of features. Notably, this
    transformation has the potential to alter total algorithm FI bar height ranking of features. """
    fracLists = []
    for each in top_fi_ave_norm_list: #each algorithm
        fracList = []
        for i in range(len(each)): #each feature
            if sum(each) == 0: #check that all feature scores are not zero to avoid zero division error
                fracList.append(0)
            else:
                fracList.append((each[i] / (sum(each))))
        fracLists.append(fracList)
    return fracLists

def weightFI(ave_metric_list,top_fi_ave_norm_list):
    """ Weights the feature importance scores by algorithm performance (intuitive because when interpreting feature importances we want to place more weight on better performing algorithms) """
    # Prepare weights
    weights = []
    # replace all balanced accuraces <=.5 with 0 (i.e. these are no better than random chance)
    for i in range(len(ave_metric_list)):
        if ave_metric_list[i] <= .5:
            ave_metric_list[i] = 0
    # normalize balanced accuracies
    for i in range(len(ave_metric_list)):
        if ave_metric_list[i] == 0:
            weights.append(0)
        else:
            weights.append((ave_metric_list[i] - 0.5) / 0.5)
    # Weight normalized feature importances
    weightedLists = []
    for i in range(len(top_fi_ave_norm_list)): #each algorithm
        weightList = np.multiply(weights[i], top_fi_ave_norm_list[i]).tolist()
        weightedLists.append(weightList)
    return weightedLists,weights

def weightFI_2(ave_metric_list,top_fi_ave_norm_list):
    """ Weights the feature importance scores by algorithm performance (intuitive because when interpreting feature importances we want to place more weight on better performing algorithms) """
    # Prepare weights
    weights = []
    for i in range(len(ave_metric_list)):
        weights.append(ave_metric_list[i])
    # Weight normalized feature importances
    weightedLists = []
    for i in range(len(top_fi_ave_norm_list)): #each algorithm
        weightList = np.multiply(weights[i], top_fi_ave_norm_list[i]).tolist()
        weightedLists.append(weightList)
    return weightedLists,weights

def weightFracFI(fracLists,weights):
    """ Weight normalized and fractionated feature importances. """
    weightedFracLists = []
    for i in range(len(fracLists)):
        weightList = np.multiply(weights[i], fracLists[i]).tolist()
        weightedFracLists.append(weightList)
    return weightedFracLists

def saveRuntime(full_path,job_start_time):
    """ Save phase runtime """
    runtime_file = open(full_path + '/runtime/runtime_Stats.txt', 'w')
    runtime_file.write(str(time.time() - job_start_time))
    runtime_file.close()

def parseRuntime(full_path,algorithms,abbrev):
    """ Loads runtime summaries from entire pipeline and parses them into a single summary file."""
    dict = {}
    for file_path in glob.glob(full_path+'/runtime/*.txt'):
        f = open(file_path,'r')
        val = float(f.readline())
        ref = file_path.split('/')[-1].split('_')[1].split('.')[0]
        if ref in abbrev:
            ref = abbrev[ref]
        if not ref in dict:
            dict[ref] = val
        else:
            dict[ref] += val
    with open(full_path+'/runtimes.csv',mode='w', newline="") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Pipeline Component","Time (sec)"])
        writer.writerow(["Exploratory Analysis",dict['exploratory']])
        writer.writerow(["Preprocessing",dict['preprocessing']])
        try:
            writer.writerow(["Mutual Information",dict['mutualinformation']])
        except:
            pass
        try:
            writer.writerow(["MultiSURF",dict['multisurf']])
        except:
            pass
        writer.writerow(["Feature Selection",dict['featureselection']])
        for algorithm in algorithms: #Report runtimes for each algorithm
            writer.writerow(([algorithm,dict[abbrev[algorithm]]]))
        writer.writerow(["Stats Summary",dict['Stats']])

def parseRuntime_2(full_path,algorithms,abbrev):
    """ Loads runtime summaries from entire pipeline and parses them into a single summary file."""
    dict = {}
    dataset_paths = os.listdir(full_path)
    removeList = removeList = ['model_evaluation','models','runtime','runtimes.csv']
    for text in removeList:
        if text in dataset_paths:
            dataset_paths.remove(text)
    with open(full_path+'/runtimes.csv',mode='w', newline="") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Pipeline Component","Time (sec)"])
        for data_dir_path in dataset_paths:
            dataset_full_path = full_path + '/' + data_dir_path
            for file_path in glob.glob(dataset_full_path+'/runtime/*.txt'):
                f = open(file_path,'r')
                val = float(f.readline())
                ref = file_path.split('/')[-1].split('_')[1].split('.')[0]
                if ref in abbrev:
                    ref = abbrev[ref]
                if not ref in dict:
                    dict[ref] = val
                else:
                    dict[ref] += val
            writer.writerow([data_dir_path+" Exploratory Analysis",dict['exploratory']])
            writer.writerow([data_dir_path+" Preprocessing",dict['preprocessing']])
            try:
                writer.writerow([data_dir_path+" Mutual Information",dict['mutualinformation']])
            except:
                pass
            try:
                writer.writerow([data_dir_path+" MultiSURF",dict['multisurf']])
            except:
                pass
        for file_path in glob.glob(full_path+'/runtime/*.txt'):
            f = open(file_path,'r')
            val = float(f.readline())
            ref = file_path.split('/')[-1].split('_')[1].split('.')[0]
            if ref in abbrev:
                ref = abbrev[ref]
            if not ref in dict:
                dict[ref] = val
            else:
                dict[ref] += val
        for algorithm in algorithms: #Repoert runtimes for each algorithm
            writer.writerow(([algorithm,dict[abbrev[algorithm]]]))
        writer.writerow(["Stats Summary",dict['Stats']])

if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],int(sys.argv[7]),sys.argv[8],sys.argv[9],sys.argv[10],int(sys.argv[11]),float(sys.argv[12]),sys.argv[13],sys.argv[14])
