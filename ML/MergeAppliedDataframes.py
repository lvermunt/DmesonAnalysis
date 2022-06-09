'''
python script to merge the applied dataframes from different childs/runlists
run: python MergeAppliedDataframes.py cfgFileNameML.yml
'''

import os
import sys
import argparse
import pickle
import yaml
import numpy as np
import pandas as pd

from hipe4ml.tree_handler import TreeHandler

def main(): #pylint: disable=too-many-statements
    # read config file
    parser = argparse.ArgumentParser(description='Arguments to pass')
    parser.add_argument('cfgFileName', metavar='text', default='cfgFileNameML.yml', help='config file name for ml')
    args = parser.parse_args()

    print('Loading analysis configuration: ...', end='\r')
    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    print('Loading analysis configuration: Done!')

    childname_list = []
    inFolderNameDataPar = inputCfg['output']['dir']
    if inFolderNameDataPar is not None:
        for subdirname in os.listdir(inFolderNameDataPar):
            if os.path.isdir(f'{inFolderNameDataPar}/{subdirname}') and "child" in subdirname:
                childname_list.append(f'{subdirname}/')
    else:
        print(f'Error: Input directory {inFolderNameDataPar} not found! Exit')
        sys.exit()

    PtBins = [[a, b] for a, b in zip(inputCfg['pt_ranges']['min'], inputCfg['pt_ranges']['max'])]
    for iBin, PtBin in enumerate(PtBins):

        arr_df_data = []
        for childname in childname_list:
            InPutDirPt = os.path.join(os.path.expanduser(inputCfg['output']['dir']), f'{childname}/pt{PtBin[0]}_{PtBin[1]}')
            filedata = f'{InPutDirPt}/Data_pT_{PtBin[0]}_{PtBin[1]}_ModelApplied.parquet.gzip'
            print('Loading unmerged data file: ', filedata, end='\r')
            DataHandler_ = TreeHandler(filedata, inputCfg['input']['treename'])
            df_ = DataHandler_.get_data_frame()
            arr_df_data.append(df_)
        print(f'\n\033[Merging applied dataframes --- {PtBin[0]} < pT < {PtBin[1]} GeV/c\033[0m')
        df_data_merge = pd.concat(arr_df_data)

        OutPutDirPt = os.path.join(os.path.expanduser(inputCfg['output']['dir']), 'merged')
        if os.path.isdir(OutPutDirPt):
            if iBin == 0:
                print((f'\033[93mWARNING: Output directory \'{OutPutDirPt}\' already exists,'
                       ' overwrites possibly ongoing!\033[0m'))
        else:
            os.makedirs(OutPutDirPt)
        OutPutDirPt = os.path.join(os.path.expanduser(inputCfg['output']['dir']), 'merged')
        df_data_merge.to_parquet(f'{OutPutDirPt}/Data_pT_{PtBin[0]}_{PtBin[1]}_ModelApplied.parquet.gzip')
        print('Merging applied dataframes: Done!')

main()
