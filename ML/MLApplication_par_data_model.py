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

def appl(inputCfg, PtBin, OutPutDirPt, ModelHandl, DataDfPtSel):
    OutputLabels = [inputCfg['output']['out_labels']['Bkg'],
                    inputCfg['output']['out_labels']['Prompt']]
    if inputCfg['output']['out_labels']['FD'] is not None:
        OutputLabels.append(inputCfg['output']['out_labels']['FD'])

    df_column_to_save_list = inputCfg['appl']['column_to_save_list']
    if not isinstance(df_column_to_save_list, list):
        print('\033[91mERROR: df_column_to_save_list must be defined!\033[0m')
        sys.exit()
    if 'inv_mass' not in df_column_to_save_list:
        print('\033[93mWARNING: inv_mass is not going to be saved in the output dataframe!\033[0m')
    if 'pt_cand' not in df_column_to_save_list:
        print('\033[93mWARNING: pt_cand is not going to be saved in the output dataframe!\033[0m')

    print('Applying ML model to data dataframe: ...', end='\r')
    yPredData = ModelHandl.predict(DataDfPtSel, inputCfg['ml']['raw_output'])
    df_column_to_save_list_data = df_column_to_save_list
    if 'pt_B' in df_column_to_save_list_data:
        df_column_to_save_list_data.remove('pt_B') # only in MC
    DataDfPtSel = DataDfPtSel.loc[:, df_column_to_save_list_data]
    if inputCfg['output']['out_labels']['FD'] is not None:
        for Pred, Lab in enumerate(OutputLabels):
            DataDfPtSel[f'ML_output_{Lab}'] = yPredData[:, Pred]
    else:
        DataDfPtSel['ML_output'] = yPredData
    DataDfPtSel.to_parquet(f'{OutPutDirPt}/Data_pT_{PtBin[0]}_{PtBin[1]}_ModelApplied.parquet.gzip')
    print('Applying ML model to data dataframe: Done!')


def main(): #pylint: disable=too-many-statements
    # read config file
    parser = argparse.ArgumentParser(description='Arguments to pass')
    parser.add_argument('cfgFileName', metavar='text', default='cfgFileNameML.yml', help='config file name for ml')
    args = parser.parse_args()

    print('Loading analysis configuration: ...', end='\r')
    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    print('Loading analysis configuration: Done!')

    #Get filename. This parallel script only works in case 'foldername' is also provided
    inFileNameData = inputCfg['input']['data']
    inFolderNameDataPar = inputCfg['input'].get('data_foldername', None)

    childname_list = []
    if inFolderNameDataPar is not None:
        #for dirpath, dirnames, filenames in os.walk(inFolderNameDataPar):
        #    for subdirname in dirnames:
        #        childname_list.append(f'{subdirname}/')
        for subdirname in os.listdir(inFolderNameDataPar):
            if os.path.isdir(f'{inFolderNameDataPar}/{subdirname}'):
                childname_list.append(f'{subdirname}/')
    else:
        print('Error: Please run the not-parallel script or provide the correct data foldername! Exit')
        sys.exit()

    for childname in childname_list:

        #if childname not in ['child_22/', 'child_23/', 'child_24/', 'child_25/', 'child_26/', 'child_27/', 'child_28/', 'child_29/', 'child_30/', 'child_31/', 'child_32/', 'child_33/', 'child_34/', 'child_35/']:
        #    print(f'Not running over {childname} for the moment (only for this test)')
        #    continue

        inFileNamesData = []
        if inFolderNameDataPar is not None:
            for dirpath, dirnames, filenames in os.walk(f'{inFolderNameDataPar}/{childname}'):
                for filename in [f for f in filenames if f.endswith(inFileNameData)]:
                    inFileNamesData.append(os.path.join(dirpath, filename))
        else:
            print('Error: Please run the not-parallel script or provide the correct data foldername! Exit')
            sys.exit()

        print(f'Loading and preparing data files {childname}: ...')

        arr_df_data2 = []
        for filedata in inFileNamesData:
            print('Loading unmerged data file: ', filedata, end='\r')
            DataHandler_ = TreeHandler(filedata, inputCfg['input']['treename'])
            df2_ = DataHandler_.get_data_frame()
            arr_df_data2.append(df2_)
        df_data_merge2 = pd.concat(arr_df_data2)
        DataHandler = TreeHandler()
        print("All data dataframes loaded...")
        DataHandler.set_data_frame(df_data_merge2)
        print("All data dataframes merged!")

        PreSelection = inputCfg['input']['preselections']
        print(f'\nTrying to apply preselections: {PreSelection}')
        if PreSelection is not None:
            DataHandler.apply_preselections(PreSelection)
            print(f'\nPreselections applied: {PreSelection}')

        PtBins = [[a, b] for a, b in zip(inputCfg['pt_ranges']['min'], inputCfg['pt_ranges']['max'])]
        DataHandler.slice_data_frame('pt_cand', PtBins, True)
        print('Loading and preparing data files: Done!')

        for iBin, PtBin in enumerate(PtBins):
            print(f'\n\033[94mStarting ML analysis --- {PtBin[0]} < pT < {PtBin[1]} GeV/c\033[0m')

            OutPutDirPt = os.path.join(os.path.expanduser(inputCfg['output']['dir']), f'{childname}/pt{PtBin[0]}_{PtBin[1]}')
            if os.path.isdir(OutPutDirPt):
                print((f'\033[93mWARNING: Output directory \'{OutPutDirPt}\' already exists,'
                       ' overwrites possibly ongoing!\033[0m'))
            else:
                os.makedirs(OutPutDirPt)

            ModelList = inputCfg['ml']['saved_models']
            ModelPath = ModelList[iBin]
            if not isinstance(ModelPath, str):
                print('\033[91mERROR: path to model not correctly defined!\033[0m')
                sys.exit()
            ModelPath = os.path.expanduser(ModelPath)
            print(f'Loaded saved model: {ModelPath}')
            #ModelHandl = ModelHandler()
            #ModelHandl.load_model_handler(ModelPath)
            model_xgb = xgb.XGBClassifier()
            model_xgb.load_model(ModelPath)
            ModelHandl = ModelHandler(model_xgb, inputCfg['ml']['training_columns'])
            appl(inputCfg, PtBin, OutPutDirPt, ModelHandl, DataHandler.get_slice(iBin))

main()
