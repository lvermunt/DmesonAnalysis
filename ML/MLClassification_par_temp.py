'''
python script to run basic training and application using the hipe4ml package
run: python MLClassification.py cfgFileNameML.yml [--train, --apply]
--train -> to perform only the training and save the models in pkl
--apply -> to perform only the application loading saved models

FORCE 4-6 TO BE PROCESSED SEPERATELY
'''
import os
import sys
import argparse
import pickle
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from hipe4ml import plot_utils
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler

from ROOT import TFile
from utilities_plot import makefill2dhist, buildbinning
from dedicated_to_root_plotting import extract_topo_distr_dstar

def data_prep(inputCfg, iBin, PtBin, OutPutDirPt, PromptDf, FDDf, BkgDf, onlyApply): #pylint: disable=too-many-statements, too-many-branches
    '''
    function for data preparation
    '''
    nPrompt = len(PromptDf)
    nFD = len(FDDf)
    nBkg = len(BkgDf)
    if FDDf.empty:
        out = f'\n     Signal: {nPrompt}\n     Bkg: {nBkg}'
    else:
        out = f'\n     Prompt: {nPrompt}\n     FD: {nFD}\n     Bkg: {nBkg}'
    print(f'Number of available candidates in {PtBin[0]} < pT < {PtBin[1]} GeV/c:{out}')

    dataset_opt = inputCfg['data_prep']['dataset_opt']
    seed_split = inputCfg['data_prep']['seed_split']
    test_f = inputCfg['data_prep']['test_fraction']
    max_cand_for_equal = inputCfg['data_prep']['max_cand_for_equal']

    if not onlyApply:
        print('Making some data preparation pltos...')
        print('IMPORTANT: The extract_topo_distr_dstar() function should not be run during training!')
        #__________________________________________________
        myfile_nsigpt = TFile.Open(f'{OutPutDirPt}/nsigma_vs_pT_{PtBin[0]}_{PtBin[1]}.root', 'RECREATE')
        myfile_nsigpt.cd()
        yname_ = ['nsigComb_Pi_0', 'nsigComb_K_0', 'nsigComb_Pi_1', 'nsigComb_K_1', 'nsigComb_Pi_2', 'nsigComb_K_2']
        xname_ = ['pt_prong0', 'pt_prong0', 'pt_prong1', 'pt_prong1', 'pt_prong2', 'pt_prong2']
        for yname, xname in zip(yname_, xname_):
            binning_nsigma = buildbinning(1, -1000, -998)
            binning_nsigma += buildbinning(2000, -100, 100)
            binning_pt = buildbinning(420, -1, 20)
            hbkg = makefill2dhist(BkgDf, f'hbkg_{yname}', binning_pt, binning_nsigma, xname, yname)
            hbkg.SetTitle(f'Background Dstar -> D0#pi;{xname};{yname}')
            hbkg.Write()
            hpr = makefill2dhist(PromptDf, f'hpr_{yname}', binning_pt, binning_nsigma, xname, yname)
            hpr.SetTitle(f'Prompt Dstar -> D0#pi;{xname};{yname}')
            hpr.Write()
            if not FDDf.empty:
                hfd = makefill2dhist(FDDf, f'hfd_{yname}', binning_pt, binning_nsigma, xname, yname)
                hfd.SetTitle(f'Feeddown Dstar -> D0#pi;{xname};{yname}')
                hfd.Write()
        myfile_nsigpt.Close()
        #__________________________________________________
        #The multiplication screws up the training
        #ONLY ENABLE WHEN YOU WANT THE PLOTS
        #topovars_dstar = ['d_len', 'd_len_xy', 'norm_dl_xy', 'cos_p', 'cos_p_xy', 'dca', 'imp_par_xy', 'max_norm_d0d0exp', 'delta_mass_D0', 'cos_t_star']
        #extract_topo_distr_dstar(PtBin, OutPutDirPt, topovars_dstar, PromptDf, FDDf, BkgDf)
        #__________________________________________________
        myfile_varcorr = TFile.Open(f'{OutPutDirPt}/varcorrs_{PtBin[0]}_{PtBin[1]}.root', 'RECREATE')
        myfile_varcorr.cd()
        xname_ = ['inv_mass', 'inv_mass', 'inv_mass']
        yname_ = ['angle_D0dkpPisoft', 'delta_mass_D0', 'pt_prong0']
        binningx_ = [buildbinning(224, 0.138, 0.250), buildbinning(224, 0.138, 0.250), buildbinning(224, 0.138, 0.250)]
        binningy_ = [buildbinning(200, 0, 0.5), buildbinning(200, 0., 100.), buildbinning(400, 0, 20)]
        for xname, yname, binx, biny in zip(xname_, yname_, binningx_, binningy_):
            hbkg = makefill2dhist(BkgDf, f'hbkg_{xname}_{yname}', binx, biny, xname, yname)
            hbkg.SetTitle(f'Background Dstar -> D0#pi;{xname};{yname}')
            hbkg.Write()
            hpr = makefill2dhist(PromptDf, f'hpr_{xname}_{yname}', binx, biny, xname, yname)
            hpr.SetTitle(f'Prompt Dstar -> D0#pi;{xname};{yname}')
            hpr.Write()
            if not FDDf.empty:
                hfd = makefill2dhist(FDDf, f'hfd_{xname}_{yname}', binx, biny, xname, yname)
                hfd.SetTitle(f'Feeddown Dstar -> D0#pi;{xname};{yname}')
                hfd.Write()
        myfile_varcorr.Close()
        #__________________________________________________

    if dataset_opt == 'equal':
        if FDDf.empty:
            nCandToKeep = min([nPrompt, nBkg, max_cand_for_equal])
            out = 'signal'
            out2 = 'signal'
        else:
            nCandToKeep = min([nPrompt, nFD, nBkg, max_cand_for_equal])
            out = 'prompt, FD'
            out2 = 'prompt'
        print((f'Keep same number of {out} and background (minimum) for training and '
               f'testing ({1 - test_f}-{test_f}): {nCandToKeep}'))
        print(f'Fraction of real data candidates used for ML: {nCandToKeep/nBkg:.5f}')

        if nPrompt > nCandToKeep:
            print((f'Remaining {out2} candidates ({nPrompt - nCandToKeep})'
                   'will be used for the efficiency together with test set'))
        if nFD > nCandToKeep:
            print((f'Remaining FD candidates ({nFD - nCandToKeep}) will be used for the '
                   'efficiency together with test set'))

        TotDf = pd.concat([BkgDf.iloc[:nCandToKeep], PromptDf.iloc[:nCandToKeep], FDDf.iloc[:nCandToKeep]], sort=True)
        if FDDf.empty:
            LabelsArray = np.array([0]*nCandToKeep + [1]*nCandToKeep)
        else:
            LabelsArray = np.array([0]*nCandToKeep + [1]*nCandToKeep + [2]*nCandToKeep)
        if test_f < 1:
            TrainSet, TestSet, yTrain, yTest = train_test_split(TotDf, LabelsArray, test_size=test_f,
                                                                random_state=seed_split)
        else:
            TrainSet = pd.DataFrame()
            TestSet = TotDf.copy()
            yTrain = pd.Series()
            yTest = LabelsArray.copy()

        TrainTestData = [TrainSet, yTrain, TestSet, yTest]
        PromptDfSelForEff = pd.concat([PromptDf.iloc[nCandToKeep:], TestSet[pd.Series(yTest).array == 1]], sort=False)
        if FDDf.empty:
            FDDfSelForEff = pd.DataFrame()
        else:
            FDDfSelForEff = pd.concat([FDDf.iloc[nCandToKeep:], TestSet[pd.Series(yTest).array == 2]], sort=False)
        del TotDf

    elif dataset_opt == 'max_signal':
        nCandBkg = round(inputCfg['data_prep']['bkg_mult'][iBin] * (nPrompt + nFD))
        out = 'signal' if FDDf.empty else 'prompt and FD'
        print((f'Keep all {out} and use {nCandBkg} bkg candidates for training and '
               f'testing ({1 - test_f}-{test_f})'))
        if nCandBkg >= nBkg:
            nCandBkg = nBkg
            print('\033[93mWARNING: using all bkg available, not good!\033[0m')
        print(f'Fraction of real data candidates used for ML: {nCandBkg/nBkg:.5f}')

        TotDf = pd.concat([BkgDf.iloc[:nCandBkg], PromptDf, FDDf], sort=True)
        if FDDf.empty:
            LabelsArray = np.array([0]*nCandBkg + [1]*nPrompt)
        else:
            LabelsArray = np.array([0]*nCandBkg + [1]*nPrompt + [2]*nFD)
        if test_f < 1:
            TrainSet, TestSet, yTrain, yTest = train_test_split(TotDf, LabelsArray, test_size=test_f,
                                                                random_state=seed_split)
        else:
            TrainSet = pd.DataFrame()
            TestSet = TotDf.copy()
            yTrain = pd.Series()
            yTest = LabelsArray.copy()

        TrainTestData = [TrainSet, yTrain, TestSet, yTest]
        PromptDfSelForEff = TestSet[pd.Series(yTest).array == 1]
        FDDfSelForEff = pd.DataFrame() if FDDf.empty else TestSet[pd.Series(yTest).array == 2]
        del TotDf

    else:
        print(f'\033[91mERROR: {dataset_opt} is not a valid option!\033[0m')
        sys.exit()

    if not onlyApply:
        # plots
        VarsToDraw = inputCfg['plots']['plotting_columns']
        LegLabels = [inputCfg['output']['leg_labels']['Bkg'],
                     inputCfg['output']['leg_labels']['Prompt']]
        if inputCfg['output']['leg_labels']['FD'] is not None:
            LegLabels.append(inputCfg['output']['leg_labels']['FD'])
        OutputLabels = [inputCfg['output']['out_labels']['Bkg'],
                        inputCfg['output']['out_labels']['Prompt']]
        if inputCfg['output']['out_labels']['FD'] is not None:
            OutputLabels.append(inputCfg['output']['out_labels']['FD'])
        ListDf = [BkgDf, PromptDf] if FDDf.empty else [BkgDf, PromptDf, FDDf]
        #_____________________________________________
        plot_utils.plot_distr(ListDf, VarsToDraw, 100, LegLabels, figsize=(12, 7),
                              alpha=0.3, log=True, grid=False, density=True)
        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
        plt.savefig(f'{OutPutDirPt}/DistributionsAll_pT_{PtBin[0]}_{PtBin[1]}.pdf')
        plt.close('all')
        #_____________________________________________
        CorrMatrixFig = plot_utils.plot_corr(ListDf, VarsToDraw, LegLabels)
        for Fig, Lab in zip(CorrMatrixFig, OutputLabels):
            plt.figure(Fig.number)
            plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9)
            Fig.savefig(f'{OutPutDirPt}/CorrMatrix{Lab}_pT_{PtBin[0]}_{PtBin[1]}.pdf')

    return TrainTestData, PromptDfSelForEff, FDDfSelForEff


def train_test(inputCfg, PtBin, OutPutDirPt, TrainTestData, iBin): #pylint: disable=too-many-statements, too-many-branches
    '''
    function for model training and testing
    '''
    n_classes = len(np.unique(TrainTestData[3]))
    modelClf = xgb.XGBClassifier(use_label_encoder=False)
    TrainCols = inputCfg['ml']['training_columns']
    HyperPars = inputCfg['ml']['hyper_par'][iBin]
    if not isinstance(TrainCols, list):
        print('\033[91mERROR: training columns must be defined!\033[0m')
        sys.exit()
    if not isinstance(HyperPars, dict):
        print('\033[91mERROR: hyper-parameters must be defined or be an empty dict!\033[0m')
        sys.exit()
    ModelHandl = ModelHandler(modelClf, TrainCols, HyperPars)

    # hyperparams optimization
    if inputCfg['ml']['hyper_par_opt']['do_hyp_opt']:
        print('Perform bayesian optimization')

        BayesOptConfig = inputCfg['ml']['hyper_par_opt']['bayes_opt_config']
        if not isinstance(BayesOptConfig, dict):
            print('\033[91mERROR: bayes_opt_config must be defined!\033[0m')
            sys.exit()

        if n_classes > 2:
            average_method = inputCfg['ml']['roc_auc_average']
            roc_method = inputCfg['ml']['roc_auc_approach']
            if not (average_method in ['macro', 'weighted'] and roc_method in ['ovo', 'ovr']):
                print('\033[91mERROR: selected ROC configuration is not valid!\033[0m')
                sys.exit()

            if average_method == 'weighted':
                metric = f'roc_auc_{roc_method}_{average_method}'
            else:
                metric = f'roc_auc_{roc_method}'
        else:
            metric = 'roc_auc'

        print('Performing hyper-parameters optimisation: ...', end='\r')
        OutFileHypPars = open(f'{OutPutDirPt}/HyperParOpt_pT_{PtBin[0]}_{PtBin[1]}.txt', 'wt')
        sys.stdout = OutFileHypPars
        ModelHandl.optimize_params_bayes(TrainTestData, BayesOptConfig, metric,
                                         nfold=inputCfg['ml']['hyper_par_opt']['nfolds'],
                                         init_points=inputCfg['ml']['hyper_par_opt']['initpoints'],
                                         n_iter=inputCfg['ml']['hyper_par_opt']['niter'],
                                         njobs=inputCfg['ml']['hyper_par_opt']['njobs'])
        OutFileHypPars.close()
        sys.stdout = sys.__stdout__
        print('Performing hyper-parameters optimisation: Done!')
        print(f'Output saved in {OutPutDirPt}/HyperParOpt_pT_{PtBin[0]}_{PtBin[1]}.txt')
        print(f'Best hyper-parameters:\n{ModelHandl.get_model_params()}')
    else:
        ModelHandl.set_model_params(HyperPars)

    # train and test the model with the updated hyper-parameters
    yPredTest = ModelHandl.train_test_model(TrainTestData, True, output_margin=inputCfg['ml']['raw_output'],
                                            average=inputCfg['ml']['roc_auc_average'],
                                            multi_class_opt=inputCfg['ml']['roc_auc_approach'])
    yPredTrain = ModelHandl.predict(TrainTestData[0], inputCfg['ml']['raw_output'])

    # save model handler in pickle
    ModelHandl.dump_model_handler(f'{OutPutDirPt}/ModelHandler_pT_{PtBin[0]}_{PtBin[1]}.pickle')
    ModelHandl.dump_original_model(f'{OutPutDirPt}/XGBoostModel_pT_{PtBin[0]}_{PtBin[1]}.model', True)

    #plots
    LegLabels = [inputCfg['output']['leg_labels']['Bkg'],
                 inputCfg['output']['leg_labels']['Prompt']]
    if inputCfg['output']['leg_labels']['FD'] is not None:
        LegLabels.append(inputCfg['output']['leg_labels']['FD'])
    OutputLabels = [inputCfg['output']['out_labels']['Bkg'],
                    inputCfg['output']['out_labels']['Prompt']]
    if inputCfg['output']['out_labels']['FD'] is not None:
        OutputLabels.append(inputCfg['output']['out_labels']['FD'])
    #_____________________________________________
    plt.rcParams["figure.figsize"] = (10, 7)
    MLOutputFig = plot_utils.plot_output_train_test(ModelHandl, TrainTestData, 80, inputCfg['ml']['raw_output'],
                                                    LegLabels, inputCfg['plots']['train_test_log'], density=True)
    if n_classes > 2:
        for Fig, Lab in zip(MLOutputFig, OutputLabels):
            Fig.savefig(f'{OutPutDirPt}/MLOutputDistr{Lab}_pT_{PtBin[0]}_{PtBin[1]}.pdf')
    else:
        MLOutputFig.savefig(f'{OutPutDirPt}/MLOutputDistr_pT_{PtBin[0]}_{PtBin[1]}.pdf')
    #_____________________________________________
    plt.rcParams["figure.figsize"] = (10, 9)
    ROCCurveFig = plot_utils.plot_roc(TrainTestData[3], yPredTest, None, LegLabels, inputCfg['ml']['roc_auc_average'],
                                      inputCfg['ml']['roc_auc_approach'])
    ROCCurveFig.savefig(f'{OutPutDirPt}/ROCCurveAll_pT_{PtBin[0]}_{PtBin[1]}.pdf')
    pickle.dump(ROCCurveFig, open(f'{OutPutDirPt}/ROCCurveAll_pT_{PtBin[0]}_{PtBin[1]}.pkl', 'wb'))
    #_____________________________________________
    plt.rcParams["figure.figsize"] = (10, 9)
    ROCCurveTTFig = plot_utils.plot_roc_train_test(TrainTestData[3], yPredTest, TrainTestData[1], yPredTrain, None,
                                                   LegLabels, inputCfg['ml']['roc_auc_average'],
                                                   inputCfg['ml']['roc_auc_approach'])
    ROCCurveTTFig.savefig(f'{OutPutDirPt}/ROCCurveTrainTest_pT_{PtBin[0]}_{PtBin[1]}.pdf')
    #_____________________________________________
    PrecisionRecallFig = plot_utils.plot_precision_recall(TrainTestData[3], yPredTest, LegLabels)
    PrecisionRecallFig.savefig(f'{OutPutDirPt}/PrecisionRecallAll_pT_{PtBin[0]}_{PtBin[1]}.pdf')
    #_____________________________________________
    plt.rcParams["figure.figsize"] = (12, 7)
    FeaturesImportanceFig = plot_utils.plot_feature_imp(TrainTestData[2][TrainCols], TrainTestData[3], ModelHandl,
                                                        LegLabels)
    n_plot = n_classes if n_classes > 2 else 1
    for iFig, Fig in enumerate(FeaturesImportanceFig):
        if iFig < n_plot:
            label = OutputLabels[iFig] if n_classes > 2 else ''
            Fig.savefig(f'{OutPutDirPt}/FeatureImportance{label}_pT_{PtBin[0]}_{PtBin[1]}.pdf')
        else:
            Fig.savefig(f'{OutPutDirPt}/FeatureImportanceAll_pT_{PtBin[0]}_{PtBin[1]}.pdf')

    return ModelHandl


def appl(inputCfg, PtBin, OutPutDirPt, ModelHandl, DataDfPtSel, PromptDfPtSelForEff, FDDfPtSelForEff):
    OutputLabels = [inputCfg['output']['out_labels']['Bkg'],
                    inputCfg['output']['out_labels']['Prompt']]
    if inputCfg['output']['out_labels']['FD'] is not None:
        OutputLabels.append(inputCfg['output']['out_labels']['FD'])
    print('Applying ML model to prompt dataframe: ...', end='\r')
    yPredPromptEff = ModelHandl.predict(PromptDfPtSelForEff, inputCfg['ml']['raw_output'])
    df_column_to_save_list = inputCfg['appl']['column_to_save_list']
    if not isinstance(df_column_to_save_list, list):
        print('\033[91mERROR: df_column_to_save_list must be defined!\033[0m')
        sys.exit()
    if 'inv_mass' not in df_column_to_save_list:
        print('\033[93mWARNING: inv_mass is not going to be saved in the output dataframe!\033[0m')
    if 'pt_cand' not in df_column_to_save_list:
        print('\033[93mWARNING: pt_cand is not going to be saved in the output dataframe!\033[0m')
    PromptDfPtSelForEff = PromptDfPtSelForEff.loc[:, df_column_to_save_list]
    if FDDfPtSelForEff.empty:
        out = 'Signal'
        PromptDfPtSelForEff['ML_output'] = yPredPromptEff
    else:
        out = 'Prompt'
        for Pred, Lab in enumerate(OutputLabels):
            PromptDfPtSelForEff[f'ML_output_{Lab}'] = yPredPromptEff[:, Pred]
    PromptDfPtSelForEff.to_parquet(f'{OutPutDirPt}/{out}_pT_{PtBin[0]}_{PtBin[1]}_ModelApplied.parquet.gzip')
    print('Applying ML model to prompt dataframe: Done!')

    if not FDDfPtSelForEff.empty:
        print('Applying ML model to FD dataframe: ...', end='\r')
        yPredFDEff = ModelHandl.predict(FDDfPtSelForEff, inputCfg['ml']['raw_output'])
        FDDfPtSelForEff = FDDfPtSelForEff.loc[:, df_column_to_save_list]
        for Pred, Lab in enumerate(OutputLabels):
            FDDfPtSelForEff[f'ML_output_{Lab}'] = yPredFDEff[:, Pred]
        FDDfPtSelForEff.to_parquet(f'{OutPutDirPt}/FD_pT_{PtBin[0]}_{PtBin[1]}_ModelApplied.parquet.gzip')
        print('Applying ML model to FD dataframe: Done!')

    print('Applying ML model to data dataframe: ...', end='\r')
    yPredData = ModelHandl.predict(DataDfPtSel, inputCfg['ml']['raw_output'])
    df_column_to_save_list_data = df_column_to_save_list
    if 'pt_B' in df_column_to_save_list_data:
        df_column_to_save_list_data.remove('pt_B') # only in MC
    DataDfPtSel = DataDfPtSel.loc[:, df_column_to_save_list_data]
    if FDDfPtSelForEff.empty:
        DataDfPtSel['ML_output'] = yPredData
    else:
        for Pred, Lab in enumerate(OutputLabels):
            DataDfPtSel[f'ML_output_{Lab}'] = yPredData[:, Pred]
    DataDfPtSel.to_parquet(f'{OutPutDirPt}/Data_pT_{PtBin[0]}_{PtBin[1]}_ModelApplied.parquet.gzip')
    print('Applying ML model to data dataframe: Done!')


def main(): #pylint: disable=too-many-statements
    # read config file
    parser = argparse.ArgumentParser(description='Arguments to pass')
    parser.add_argument('cfgFileName', metavar='text', default='cfgFileNameML.yml', help='config file name for ml')
    parser.add_argument("--train", help="perform only training and testing", action="store_true")
    parser.add_argument("--apply", help="perform only application", action="store_true")
    args = parser.parse_args()

    print('Loading analysis configuration: ...', end='\r')
    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    print('Loading analysis configuration: Done!')

    #Get filename. This parallel script only works in case 'foldername' is also provided
    inFileNamePrompt = inputCfg['input']['prompt']
    inFileNameFD = inputCfg['input']['FD']
    inFileNameData = inputCfg['input']['data']
    inFileNamesPrompt = []
    inFileNamesFD = []
    inFileNamesData = []

    inFolderNamePromptPar = inputCfg['input'].get('prompt_foldername', None)
    inFolderNameFDPar = inputCfg['input'].get('FD_foldername', None)
    inFolderNameDataPar = inputCfg['input'].get('data_foldername', None)
    if inFolderNamePromptPar is not None:
        for dirpath, dirnames, filenames in os.walk(inFolderNamePromptPar):
            for filename in [f for f in filenames if f.endswith(inFileNamePrompt)]:
                inFileNamesPrompt.append(os.path.join(dirpath, filename))
        print("Check if correct folders are loaded prompt: ")
        print(inFileNamesPrompt)
    else:
        print('Error: Please run the not-parallel script or provide the correct prompt foldername! Exit')
        sys.exit()

    if inFolderNameFDPar is not None:
        for dirpath, dirnames, filenames in os.walk(inFolderNameFDPar):
            for filename in [f for f in filenames if f.endswith(inFileNameFD)]:
                inFileNamesFD.append(os.path.join(dirpath, filename))
        print("Check if correct folders are loaded FD: ")
        print(inFileNamesFD)
    else:
        print('Error: Please run the not-parallel script or provide the correct FD foldername! Exit')
        sys.exit()

    if inFolderNameDataPar is not None:
        if isinstance(inFolderNameDataPar, list):
            for il in range(len(inFolderNameDataPar)):
                for dirpath, dirnames, filenames in os.walk(inFolderNameDataPar[il]):
                    for filename in [f for f in filenames if f.endswith(inFileNameData)]:
                        inFileNamesData.append(os.path.join(dirpath, filename))
        else:
            for dirpath, dirnames, filenames in os.walk(inFolderNameDataPar):
                for filename in [f for f in filenames if f.endswith(inFileNameData)]:
                    inFileNamesData.append(os.path.join(dirpath, filename))
    else:
        print('Error: Please run the not-parallel script or provide the correct data foldername! Exit')
        sys.exit()

    process_sep_query = inputCfg['data_prep'].get('proc_sep_query', ['pt_cand >= 0'])
    process_sep_frac = inputCfg['data_prep'].get('proc_sep_frac', [1])
    process_sep_ptbin = inputCfg['data_prep'].get('proc_sep_ptbin', [[True] * len(inputCfg['pt_ranges']['min'])])

    for frac_, query_, ptbin_ in zip(process_sep_frac, process_sep_query, process_sep_ptbin):
        #print('Loading and preparing data files: ...', end='\r')
        print('Loading and preparing data files: ...')
        arr_df_prompt = []
        for fileprompt in inFileNamesPrompt:
            PromptHandler_ = TreeHandler(fileprompt, inputCfg['input']['treename'])
            df_ = PromptHandler_.get_data_frame()
            df_ = df_.query(query_)
            arr_df_prompt.append(df_)
        df_prompt_merge = pd.concat(arr_df_prompt)
        PromptHandler = TreeHandler()
        print("All prompt dataframes loaded...")
        PromptHandler.set_data_frame(df_prompt_merge)
        print("All prompt dataframes merged!")

        if inputCfg['input']['FD'] is None:
            FDHandler = None
        else:
            arr_df_fd = []
            for filefd in inFileNamesFD:
                FDHandler_ = TreeHandler(filefd, inputCfg['input']['treename'])
                df_ = FDHandler_.get_data_frame()
                df_ = df_.query(query_)
                arr_df_fd.append(df_)
            df_fd_merge = pd.concat(arr_df_fd)
            FDHandler = TreeHandler()
            print("All FD dataframes loaded...")
            FDHandler.set_data_frame(df_fd_merge)
            print("All FD dataframes merged!")

        arr_df_data = []
        arr_df_data2 = []
        for filedata in inFileNamesData:
            print('Loading unmerged data file: ', filedata, end='\r')
            DataHandler_ = TreeHandler(filedata, inputCfg['input']['treename'])
            if inputCfg['data_prep']['filt_bkg_mass']:
                BkgHandler_ = DataHandler_.get_subset(inputCfg['data_prep']['filt_bkg_mass'], frac=frac_,
                                                      rndm_state=inputCfg['data_prep']['seed_split'])
            else:
                BkgHandler_ = DataHandler_
            df_ = BkgHandler_.get_data_frame()
            df_ = df_.query(query_)
            arr_df_data.append(df_)
            df2_ = DataHandler_.get_data_frame()
            df2_ = df2_.query(query_)
            df2_ = df2_.iloc[:int(frac_*len(df2_))]
            arr_df_data2.append(df2_)
        df_data_merge = pd.concat(arr_df_data)
        df_data_merge2 = pd.concat(arr_df_data2)
        BkgHandler = TreeHandler()
        DataHandler = TreeHandler()
        print("All data dataframes loaded...")
        BkgHandler.set_data_frame(df_data_merge)
        DataHandler.set_data_frame(df_data_merge2)
        print("All data dataframes merged!")

        PreSelection = inputCfg['input']['preselections']
        print(f'\nTrying to apply preselections: {PreSelection}')
        if PreSelection is not None:
            PromptHandler.apply_preselections(PreSelection)
            if FDHandler is not None:
                FDHandler.apply_preselections(PreSelection)
            DataHandler.apply_preselections(PreSelection)
            BkgHandler.apply_preselections(PreSelection)
            print(f'\nPreselections applied: {PreSelection}')

        PtBins = [[a, b] for a, b in zip(inputCfg['pt_ranges']['min'], inputCfg['pt_ranges']['max'])]
        PromptHandler.slice_data_frame('pt_cand', PtBins, True)
        if FDHandler is not None:
            FDHandler.slice_data_frame('pt_cand', PtBins, True)
        DataHandler.slice_data_frame('pt_cand', PtBins, True)
        BkgHandler.slice_data_frame('pt_cand', PtBins, True)
        print('Loading and preparing data files: Done!')

        for iBin, PtBin in enumerate(PtBins):
            if not ptbin_[iBin]:
                continue
            print(f'\n\033[94mStarting ML analysis --- {PtBin[0]} < pT < {PtBin[1]} GeV/c\033[0m')

            OutPutDirPt = os.path.join(os.path.expanduser(inputCfg['output']['dir']), f'pt{PtBin[0]}_{PtBin[1]}')
            if args.apply and inputCfg['output'].get('dirapplyonly', False):
                OutPutDirPt = os.path.join(os.path.expanduser(inputCfg['output']['dirapplyonly']), f'pt{PtBin[0]}_{PtBin[1]}')
            if os.path.isdir(OutPutDirPt):
                print((f'\033[93mWARNING: Output directory \'{OutPutDirPt}\' already exists,'
                       ' overwrites possibly ongoing!\033[0m'))
            else:
                os.makedirs(OutPutDirPt)

            # data preparation
            #_____________________________________________
            FDDfPt = pd.DataFrame() if FDHandler is None else FDHandler.get_slice(iBin)
            TrainTestData, PromptDfSelForEff, FDDfSelForEff = data_prep(inputCfg, iBin, PtBin, OutPutDirPt,
                                                                        PromptHandler.get_slice(iBin), FDDfPt,
                                                                        BkgHandler.get_slice(iBin), args.apply)
            if args.apply and inputCfg['data_prep']['test_fraction'] < 1.:
                print('\033[93mWARNING: Using only a fraction of the MC for the application! Are you sure?\033[0m')

            # training, testing
            #_____________________________________________
            if not args.apply:
                ModelHandl = train_test(inputCfg, PtBin, OutPutDirPt, TrainTestData, iBin)
            else:
                ModelList = inputCfg['ml']['saved_models']
                ModelPath = ModelList[iBin]
                if not isinstance(ModelPath, str):
                    print('\033[91mERROR: path to model not correctly defined!\033[0m')
                    sys.exit()
                ModelPath = os.path.expanduser(ModelPath)
                print(f'Loaded saved model: {ModelPath}')
                ModelHandl = ModelHandler()
                ModelHandl.load_model_handler(ModelPath)

            # model application
            #_____________________________________________
            if not args.train:
                appl(inputCfg, PtBin, OutPutDirPt, ModelHandl, DataHandler.get_slice(iBin),
                     PromptDfSelForEff, FDDfSelForEff)

            # delete dataframes to release memory
            for data in TrainTestData:
                del data
            del PromptDfSelForEff, FDDfSelForEff

main()
