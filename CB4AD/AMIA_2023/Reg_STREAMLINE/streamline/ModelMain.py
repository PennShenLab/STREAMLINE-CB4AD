"""
File: ModelMain.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 5 of STREAMLINE - This 'Main' script manages Phase 5 run parameters, updates the metadata file (with user specified run parameters across pipeline run)
             and submits job to run locally (to run serially) or on a linux computing cluster (parallelized).  This script runs ModelJob.py which conducts machine learning
             modeling using respective training datasets. This pipeline currently includes the following 13 ML modeling algorithms for binary classification:
             * Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LGBoost, Support Vector Machine (SVM), Artificial Neural Network (ANN),
             * k Nearest Neighbors (k-NN), Educational Learning Classifier System (eLCS), X Classifier System (XCS), and the Extended Supervised Tracking and Classifying System (ExSTraCS)
             This phase includes hyperparameter optimization of all algorithms (other than naive bayes), model training, model feature importance estimation (using internal algorithm
             estimations, if available, or via permutation feature importance), and performance evaluation on hold out testing data. This script creates a single job for each
             combination of cv dataset (for each original target dataset) and ML modeling algorithm. In addition to an option to check the completion of all jobs, this script also has a
             'resubmit' option that will run any jobs that may have failed from a previous run. All 'Main' scripts in this pipeline have the potential to be extended by users to
             submit jobs to other parallel computing frameworks (e.g. cloud computing).
Warnings: Designed to be run following the completion of STREAMLINE Phase 4 (FeatureSelectionMain.py). SVM modeling should only be applied when data scaling is applied by the pipeline
            Logistic Regression' baseline model feature importance estimation is determined by the exponential of the feature's coefficient. This should only be used if data scaling is
            applied by the pipeline. Otherwise 'use_uniform_FI' should be True.
Sample Run Command (Linux cluster parallelized with all default run parameters):
    python ModelMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1
Sample Run Command (Local/serial with with all default run parameters):
    python ModelMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1 --run-parallel False
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import argparse
import os
import sys
import pandas as pd
import glob
import ModelJob
import time
import csv
import random
import shutil
import pickle

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    #No defaults
    parser.add_argument('--out-path', dest='output_path', type=str, help='path to output directory', default = '/Users/yanbo/Dropbox/STREAMLINE-Regression_AMIA/Colab_Output')
    parser.add_argument('--exp-name', dest='experiment_name', type=str, help='name of experiment (no spaces)', default = 'Demo_Experiment')
    #Sets default run all or none to make algorithm selection from command line simpler
    parser.add_argument('--do-all', dest='do_all', type=str, help='run all modeling algorithms by default (when set False, individual algorithms are activated individually)',default='False')
    parser.add_argument('--do-CommonReg', dest='do_CommonReg', type=str, help='run common regression algorithms by default (when set False, common regression algorithms are muted)',default='True')
    parser.add_argument('--do-L21Series', dest='do_L21Series', type=str, help='run L21 regression algorithms (when set False, L21 regression algorithms are muted)',default='True')
    #ML modeling algorithms: Defaults available
    parser.add_argument('--do-linReg', dest='do_linReg', type=str, help='run linear regression modeling',default='True')
    parser.add_argument('--do-ENReg', dest='do_ENReg', type=str, help='run elasticnet regression modeling',default='True')
    parser.add_argument('--do-RFReg', dest='do_RFReg', type=str, help='run random forest regressor modeling',default='False')
    parser.add_argument('--do-AdaReg', dest='do_AdaReg', type=str, help='run adaboost regressor modeling',default='False')
    parser.add_argument('--do-GradReg', dest='do_GradReg', type=str, help='run gradientboosting regressor modeling',default='False')
    parser.add_argument('--do-SVR', dest='do_SVR', type=str, help='run support vector regression modeling',default='True')
    parser.add_argument('--do-GL', dest='do_GL', type=str, help='run group lasso regressor modeling',default='False')
    #L21-series models
    parser.add_argument('--do-L21Reg', dest='do_L21Reg', type=str, help='run L21 Norm regression modeling',default='True')
    parser.add_argument('--do-L21GMMReg', dest='do_L21GMMReg', type=str, help='run GMM L21 Norm regression modeling',default='False')
    parser.add_argument('--do-L21DGMMReg', dest='do_L21DGMMReg', type=str, help='run DGMM L21 Norm regression modeling',default='False')
    ### Add new algorithms here...
    #Group Lasso Parameters - Defaults available
    parser.add_argument('--groups-path', dest='groups_path', type=str, help='path to defined groups file', default = '/Users/yanbo/Dropbox/STREAMLINE-Regression_AMIA/streamline/groups.csv')
    #Other Analysis Parameters - Defaults available
    parser.add_argument('--metric', dest='primary_metric', type=str,help='primary scikit-learn specified scoring metric used for hyperparameter optimization and permutation-based model feature importance evaluation', default='explained_variance')
    parser.add_argument('--subsample', dest='training_subsample', type=int, help='for long running algos, option to subsample training set (0 for no subsample)', default=0)
    parser.add_argument('--use-uniformFI', dest='use_uniform_FI', type=str, help='overrides use of any available feature importance estimate methods from models, instead using permutation_importance uniformly',default='True')
    #Hyperparameter sweep options - Defaults available
    parser.add_argument('--n-trials', dest='n_trials', type=str,help='# of bayesian hyperparameter optimization trials using optuna (specify an integer or None)', default=50)
    parser.add_argument('--timeout', dest='timeout', type=str,help='seconds until hyperparameter sweep stops running new trials (Note: it may run longer to finish last trial started) If set to None, STREAMLINE is completely replicable, but will take longer to run', default=900) #900 sec = 15 minutes default
    parser.add_argument('--export-hyper-sweep', dest='export_hyper_sweep_plots', type=str, help='export optuna-generated hyperparameter sweep plots', default='True')
    #Lostistical arguments - Defaults available
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel on LSF compatible computing cluster',default="False")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of parallel computing queue (uses our research groups queue by default)',default="i2c2_normal")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')
    parser.add_argument('-r','--do-resubmit',dest='do_resubmit', help='Boolean: Rerun any jobs that did not complete (or failed) in an earlier run.', action='store_true')

    options = parser.parse_args(argv[1:])
    job_counter = 0

    # Argument checks
    if not os.path.exists(options.output_path):
        raise Exception("Output path must exist (from phase 1) before phase 5 can begin")
    if not os.path.exists(options.output_path + '/' + options.experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 5 can begin")

    #Unpickle metadata from previous phase
    file = open(options.output_path+'/'+options.experiment_name+'/'+"metadata.pickle", 'rb')
    metadata = pickle.load(file)
    file.close()
    #Load variables specified earlier in the pipeline from metadata
    class_label = metadata['Class Label']
    instance_label = metadata['Instance Label']
    random_state = int(metadata['Random Seed'])
    cv_partitions = int(metadata['CV Partitions'])
    filter_poor_features = metadata['Filter Poor Features']
    jupyterRun = metadata['Run From Jupyter Notebook']

    if options.do_resubmit: #Attempts to resolve optuna hyperparameter optimization hangup (i.e. when it runs indefinitely for a given random seed attempt)
        random_state = random.randint(1,1000)

    #Create ML modeling algorithm information dictionary, given as ['algorithm used (set to true initially by default)','algorithm abreviation', 'color used for algorithm on figures']
    ### Note that other named colors used by matplotlib can be found here: https://matplotlib.org/3.5.0/_images/sphx_glr_named_colors_003.png
    ### Make sure new ML algorithm abbreviations and color designations are unique
    algInfo = {}
    algInfo['Linear Regression'] = [True,'Linear Regression','red']
    algInfo['Elastic Net'] = [True, 'Elastic Net', 'steelblue']
    algInfo['Group Lasso'] = [True, 'Group Lasso', 'orange']
    algInfo['L21Reg'] = [True,'L21Reg','green']
    algInfo['L21GMMReg'] = [True, 'L21GMMReg', 'darkslategray']
    algInfo['L21DGMMReg'] = [True, 'L21DGMMReg', 'magenta']
    algInfo['RF Regressor'] = [True, 'RF Regressor', 'navy']
    algInfo['AdaBoost'] = [True, 'AdaBoost', 'teal']
    algInfo['GradBoost'] = [True, 'GradBoost', 'olive']
    algInfo['SVR'] = [True, 'SVR', 'rosybrown']
    ### Add new algorithms here...

    #Set up ML algorithm True/False use
    if not eval(options.do_all): #If do all algorithms is false
        for key in algInfo:
            algInfo[key][0] = False #Set algorithm use to False

    #Set algorithm use truth for each algorithm specified by user (i.e. if user specified True/False for a specific algorithm)
    if not options.do_linReg == 'None':
        algInfo['Linear Regression'][0] = eval(options.do_linReg)
    if not options.do_ENReg == 'None':
        algInfo['Elastic Net'][0] = eval(options.do_ENReg)
    if not options.do_GL == 'None':
        algInfo['Group Lasso'][0] = eval(options.do_GL)
    if not options.do_RFReg == 'None':
        algInfo['RF Regressor'][0] = eval(options.do_RFReg)
    if not options.do_AdaReg == 'None':
        algInfo['AdaBoost'][0] = eval(options.do_AdaReg)
    if not options.do_GradReg == 'None':
        algInfo['GradBoost'][0] = eval(options.do_GradReg)
    if not options.do_SVR == 'None':
        algInfo['SVR'][0] = eval(options.do_SVR)
    if not options.do_L21Reg == 'None':
        algInfo['L21Reg'][0] = eval(options.do_L21Reg)
    if not options.do_L21GMMReg == 'None':
        algInfo['L21GMMReg'][0] = eval(options.do_L21GMMReg)
    if not options.do_L21DGMMReg == 'None':
        algInfo['L21DGMMReg'][0] = eval(options.do_L21DGMMReg)

    ### Add new algorithms here...

    #Pickle the algorithm information dictionary for future use
    pickle_out = open(options.output_path+'/'+options.experiment_name+'/'+"algInfo.pickle", 'wb')
    pickle.dump(algInfo,pickle_out)
    pickle_out.close()
    
    if options.do_L21Series == 'True':
        dataset_paths = os.listdir(options.output_path+"/"+options.experiment_name)
        removeList = ['metadata.pickle','metadata.csv','algInfo.pickle','jobsCompleted','logs','jobs','DatasetComparisons','UsefulNotebooks',options.experiment_name+'_ML_Pipeline_Report.pdf']
        for text in removeList:
            if text in dataset_paths:
                dataset_paths.remove(text)
        path_1 = options.output_path + "/" + options.experiment_name
        os.mkdir(path_1+'/L21')
        for dataset_directory_path_1 in dataset_paths:
            dest_dir = path_1+'/L21'+'/'+dataset_directory_path_1
            src_dir = path_1+'/'+dataset_directory_path_1
            print(dest_dir)
            print(src_dir)
            shutil.copytree(src_dir, dest_dir)
    
    #Make list of algorithms to be run (full names)
    algorithms = []
    for key in algInfo:
        if algInfo[key][0]: #Algorithm is true
            algorithms.append(key)

    if not options.do_check and not options.do_resubmit: #Run job submission
        dataset_paths = os.listdir(options.output_path + "/" + options.experiment_name)
        removeList = ['L21', 'metadata.pickle','metadata.csv','algInfo.pickle','jobsCompleted','logs','jobs','DatasetComparisons',options.experiment_name+'_ML_Pipeline_Report.pdf']
        for text in removeList:
            if text in dataset_paths:
                dataset_paths.remove(text)
        if options.do_CommonReg == 'True':
          for dataset_directory_path in dataset_paths:
            full_path = options.output_path + "/" + options.experiment_name + "/" + dataset_directory_path
            if not os.path.exists(full_path+'/models'):
                os.mkdir(full_path+'/models')
            if not os.path.exists(full_path+'/model_evaluation'):
                os.mkdir(full_path+'/model_evaluation')
            if not os.path.exists(full_path+'/models/pickledModels'):
                os.mkdir(full_path+'/models/pickledModels')
            for cvCount in range(cv_partitions):
                train_file_path = full_path+'/CVDatasets/'+dataset_directory_path+"_CV_"+str(cvCount)+"_Train.csv"
                test_file_path = full_path + '/CVDatasets/' + dataset_directory_path + "_CV_" + str(cvCount) + "_Test.csv"
                for algorithm in algorithms:
                  if algorithm != 'L21Reg' and algorithm !='L21GMMReg' and algorithm !='L21DGMMReg':
                    algAbrev = algInfo[algorithm][1]
                    algNoSpace = algorithm.replace(" ", "_")
                    job_counter += 1
                    if eval(options.run_parallel):
                        submitClusterJob(algNoSpace,train_file_path,test_file_path,full_path,options.n_trials,options.timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,options.output_path+'/'+options.experiment_name,cvCount,filter_poor_features,options.reserved_memory,options.maximum_memory,options.training_subsample,options.queue,options.use_uniform_FI,options.primary_metric,algAbrev,options.groups_path, jupyterRun)
                    else:
                        submitLocalJob(algNoSpace,train_file_path,test_file_path,full_path,options.n_trials,options.timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,options.training_subsample,options.use_uniform_FI,options.primary_metric,algAbrev,options.groups_path,jupyterRun)
        if options.do_L21Series == 'True':
          full_path = options.output_path + "/" + options.experiment_name + "/" + 'L21'
          for cvCount in range(cv_partitions):
                dataset_paths_2 = dataset_paths.copy()
                train_file_path_1 = full_path+'/'+dataset_paths[0]+'/CVDatasets/'+dataset_paths_2[0]+"_CV_"+str(cvCount)+"_Train.csv"
                test_file_path_1 = full_path+'/'+dataset_paths[0]+ '/CVDatasets/' + dataset_paths_2[0] + "_CV_" + str(cvCount) + "_Test.csv"
                train_file_path_2 = full_path+'/'+dataset_paths[1]+'/CVDatasets/'+dataset_paths_2[1]+"_CV_"+str(cvCount)+"_Train.csv"
                test_file_path_2 = full_path +'/'+dataset_paths[1]+ '/CVDatasets/' + dataset_paths_2[1] + "_CV_" + str(cvCount) + "_Test.csv"
                train_file_path_3 = full_path+'/'+dataset_paths[2]+'/CVDatasets/'+dataset_paths_2[2]+"_CV_"+str(cvCount)+"_Train.csv"
                test_file_path_3 = full_path +'/'+dataset_paths[2]+ '/CVDatasets/' + dataset_paths_2[2] + "_CV_" + str(cvCount) + "_Test.csv"
                for algorithm in algorithms:
                    if algorithm == 'L21Reg' or algorithm =='L21GMMReg' or algorithm =='L21DGMMReg':
                        algAbrev = algInfo[algorithm][1]
                        algNoSpace = algorithm.replace(" ", "_")
                        job_counter += 1
                        if eval(options.run_parallel):
                            submitClusterJob(algNoSpace,train_file_path,test_file_path,full_path,options.n_trials,options.timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,options.output_path+'/'+options.experiment_name,cvCount,filter_poor_features,options.reserved_memory,options.maximum_memory,options.training_subsample,options.queue,options.use_uniform_FI,options.primary_metric,algAbrev,options.groups_path, jupyterRun)
                        else:
                            submitLocalJob_2(algNoSpace,train_file_path_1, train_file_path_2, train_file_path_3,test_file_path_1, test_file_path_2, test_file_path_3,full_path,options.n_trials,options.timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,options.training_subsample,options.use_uniform_FI,options.primary_metric,algAbrev, options.output_path, options.experiment_name, jupyterRun)
#                    print(algorithm)
#                    algAbrev = algInfo[algorithm][1]
#                    #Get header names for current CV dataset for use later in GP tree visulaization
#                    data_name = full_path.split('/')[-1]
#                    feature_names = pd.read_csv(full_path+'/'+dataset_paths[0]+'/CVDatasets/'+dataset_paths_2[0]+'_CV_'+str(cvCount)+'_Test.csv').columns.values.tolist()
#                    if instance_label != 'None':
#                        feature_names.remove(instance_label)
#                    feature_names.remove(class_label)
#                    #Get hyperparameter grid
#                    param_grid = hyperparameters(random_state,feature_names)[algorithm]
#                    ModelJob.runModel_2(algorithm,train_file_path_1, train_file_path_2, train_file_path_3,test_file_path_1, test_file_path_2, test_file_path_3,full_path,n_trials,timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,training_subsample,use_uniform_FI,primary_metric,param_grid,algAbrev, output_path, experiment_name)

                    
        #Update metadata
        metadata['Linear Regression'] = str(algInfo['Linear Regression'][0])
        metadata['Elastic Net'] = str(algInfo['Elastic Net'][0])
        metadata['Group Lasso'] = str(algInfo['Group Lasso'][0])
        metadata['RF Regressor'] = str(algInfo['RF Regressor'][0])
        metadata['AdaBoost'] = str(algInfo['AdaBoost'][0])
        metadata['GradBoost'] = str(algInfo['GradBoost'][0])
        metadata['SVR'] = str(algInfo['SVR'][0])
        metadata['L21Reg'] = str(algInfo['L21Reg'][0])
        metadata['L21GMMReg'] = str(algInfo['L21GMMReg'][0])
        metadata['L21DGMMReg'] = str(algInfo['L21DGMMReg'][0])
        ### Add new algorithms here...

        metadata['Primary Metric'] = options.primary_metric
        metadata['Uniform Feature Importance Estimation (Models)'] = options.use_uniform_FI
        metadata['Hyperparameter Sweep Number of Trials'] = options.n_trials
        metadata['Hyperparameter Timeout'] = options.timeout
        metadata['Export Hyperparameter Sweep Plots'] = options.export_hyper_sweep_plots
        #Pickle the metadata for future use
        pickle_out = open(options.output_path+'/'+options.experiment_name+'/'+"metadata.pickle", 'wb')
        pickle.dump(metadata,pickle_out)
        pickle_out.close()

    elif options.do_check and not options.do_resubmit: #run job completion checks
        datasets = os.listdir(options.output_path + "/" + options.experiment_name)
        removeList = removeList = ['L21', 'metadata.pickle','metadata.csv','algInfo.pickle','jobsCompleted','logs','jobs','DatasetComparisons','UsefulNotebooks']
        for text in removeList:
            if text in datasets:
                datasets.remove(text)

        phase5Jobs = []
        for dataset in datasets:
            for cv in range(cv_partitions):
                for algorithm in algorithms:
                    phase5Jobs.append('job_model_' + dataset + '_' + str(cv) +'_' +algInfo[algorithm][1]+'.txt') #use algorithm abreviation for filenames

        for filename in glob.glob(options.output_path + "/" + options.experiment_name + '/jobsCompleted/job_model*'):
            ref = filename.split('/')[-1]
            phase5Jobs.remove(ref)
        for job in phase5Jobs:
            print(job)
        if len(phase5Jobs) == 0:
            print("All Phase 5 Jobs Completed")
        else:
            print("Above Phase 5 Jobs Not Completed")
        print()

    elif options.do_resubmit and not options.do_check: #resubmit any jobs that didn't finish in previous run (mix of job check and job submit)
        datasets = os.listdir(options.output_path + "/" + options.experiment_name)
        removeList = removeList = ['L21', 'metadata.pickle','metadata.csv','algInfo.pickle','jobsCompleted','logs','jobs','DatasetComparisons','UsefulNotebooks']
        for text in removeList:
            if text in datasets:
                datasets.remove(text)

        #start by making list of finished jobs instead of all jobs then step through loop
        phase5completed = []
        for filename in glob.glob(options.output_path + "/" + options.experiment_name + '/jobsCompleted/job_model*'):
            ref = filename.split('/')[-1]
            phase5completed.append(ref)
        if options.do_CommonReg == 'True':
          for dataset in datasets:
            for cv in range(cv_partitions):
                for algorithm in algorithms:
                  if algorithm != 'L21Reg' and algorithm !='L21GMMReg' and algorithm !='L21DGMMReg':
                    algAbrev = algInfo[algorithm][1]
                    algNoSpace = algorithm.replace(" ", "_")
                    targetFile = 'job_model_' + dataset + '_' + str(cv) +'_' +algInfo[algorithm][1]+'.txt'
                    if targetFile not in phase5completed: #target for a re-submit
                        full_path = options.output_path + "/" + options.experiment_name + "/" + dataset
                        train_file_path = full_path+'/CVDatasets/'+dataset+"_CV_"+str(cv)+"_Train.csv"
                        test_file_path = full_path + '/CVDatasets/' + dataset + "_CV_" + str(cv) + "_Test.csv"
                        if eval(options.run_parallel):
                            job_counter += 1
                            submitClusterJob(algNoSpace,train_file_path,test_file_path,full_path,options.n_trials,options.timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,options.output_path+'/'+options.experiment_name,cv,filter_poor_features,options.reserved_memory,options.maximum_memory,options.training_subsample,options.queue,options.use_uniform_FI,options.primary_metric,algAbrev,options.groups_path,jupyterRun)
                        else:
                            submitLocalJob(algNoSpace,train_file_path,test_file_path,full_path,options.n_trials,options.timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,cv,filter_poor_features,options.training_subsample,options.use_uniform_FI,options.primary_metric,algAbrev,options.groups_path,jupyterRun)
                    
        if options.do_L21Series == 'True':
          full_path = options.output_path + "/" + options.experiment_name + "/" + 'L21'
          for cvCount in range(cv_partitions):
                dataset_paths_2 = dataset_paths.copy()
                train_file_path_1 = full_path+'/'+dataset_paths[0]+'/CVDatasets/'+dataset_paths_2[0]+"_CV_"+str(cvCount)+"_Train.csv"
                test_file_path_1 = full_path+'/'+dataset_paths[0]+ '/CVDatasets/' + dataset_paths_2[0] + "_CV_" + str(cvCount) + "_Test.csv"
                train_file_path_2 = full_path+'/'+dataset_paths[1]+'/CVDatasets/'+dataset_paths_2[1]+"_CV_"+str(cvCount)+"_Train.csv"
                test_file_path_2 = full_path +'/'+dataset_paths[1]+ '/CVDatasets/' + dataset_paths_2[1] + "_CV_" + str(cvCount) + "_Test.csv"
                train_file_path_3 = full_path+'/'+dataset_paths[2]+'/CVDatasets/'+dataset_paths_2[2]+"_CV_"+str(cvCount)+"_Train.csv"
                test_file_path_3 = full_path +'/'+dataset_paths[2]+ '/CVDatasets/' + dataset_paths_2[2] + "_CV_" + str(cvCount) + "_Test.csv"
                for algorithm in algorithms:
                    if algorithm == 'L21Reg' or algorithm =='L21GMMReg' or algorithm =='L21DGMMReg':
                        algAbrev = algInfo[algorithm][1]
                        algNoSpace = algorithm.replace(" ", "_")
                        job_counter += 1
                        if eval(options.run_parallel):
                            submitClusterJob(algNoSpace,train_file_path,test_file_path,full_path,options.n_trials,options.timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,options.output_path+'/'+options.experiment_name,cvCount,filter_poor_features,options.reserved_memory,options.maximum_memory,options.training_subsample,options.queue,options.use_uniform_FI,options.primary_metric,algAbrev,options.groups_path, jupyterRun)
                        else:
                            submitLocalJob_2(algNoSpace,train_file_path_1, train_file_path_2, train_file_path_3,test_file_path_1, test_file_path_2, test_file_path_3,full_path,options.n_trials,options.timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,options.training_subsample,options.use_uniform_FI,options.primary_metric,algAbrev, options.output_path, options.experiment_name, jupyterRun)
        
    else:
        print("Run options in conflict. Do not request to run check and resubmit at the same time.")

    if not options.do_check:
        print(str(job_counter)+ " jobs submitted in Phase 5")

def submitLocalJob(algNoSpace,train_file_path,test_file_path,full_path,n_trials,timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,training_subsample,use_uniform_FI,primary_metric,algAbrev,groups_path,jupyterRun):
    """ Runs ModelJob.py locally, once for each combination of cv dataset (for each original target dataset) and ML modeling algorithm. These runs will be completed serially rather than in parallel. """
    ModelJob.job(algNoSpace,train_file_path,test_file_path,full_path,n_trials,timeout,export_hyper_sweep_plots, instance_label,class_label,random_state,cvCount,filter_poor_features,training_subsample,use_uniform_FI,primary_metric,algAbrev, groups_path, jupyterRun)
    
def submitLocalJob_2(algNoSpace,train_file_path_1, train_file_path_2, train_file_path_3,test_file_path_1, test_file_path_2, test_file_path_3,full_path,n_trials,timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,training_subsample,use_uniform_FI,primary_metric,algAbrev, output_path, experiment_name, jupyterRun):
    """ Runs ModelJob.py locally, once for each combination of cv dataset (for each original target dataset) and ML modeling algorithm. These runs will be completed serially rather than in parallel. """
    ModelJob.job_2(algNoSpace,train_file_path_1, train_file_path_2, train_file_path_3,test_file_path_1, test_file_path_2, test_file_path_3,full_path,n_trials,timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,training_subsample,use_uniform_FI,primary_metric,algAbrev, output_path, experiment_name, jupyterRun)
    
def submitClusterJob(algNoSpace,train_file_path,test_file_path,full_path,n_trials,timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,experiment_path,cvCount,filter_poor_features,reserved_memory,maximum_memory,training_subsample,queue,use_uniform_FI,primary_metric,algAbrev,groups_path,jupyterRun):
    """ Runs ModelJob.py once for each combination of cv dataset (for each original target dataset) and ML modeling algorithm. Runs in parallel on a linux-based computing cluster that uses an IBM Spectrum LSF for job scheduling."""
    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/P5_'+str(algAbrev)+'_'+str(cvCount)+'_'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q '+queue+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/P5_'+str(algAbrev)+'_'+str(cvCount)+'_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/P5_'+str(algAbrev)+'_'+str(cvCount)+'_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/ModelJob.py '+algNoSpace+" "+train_file_path+" "+test_file_path+" "+full_path+" "+
                  str(n_trials)+" "+str(timeout)+" "+export_hyper_sweep_plots+" "+instance_label+" "+class_label+" "+
                  str(random_state)+" "+str(cvCount)+" "+str(filter_poor_features)+" "+str(training_subsample)+" "+str(use_uniform_FI)+" "+str(primary_metric)+" "+str(algAbrev)+" "+str(groups_path)+" "+str(jupyterRun)+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
