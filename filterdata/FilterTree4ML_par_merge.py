'''
python script to only do the all-into-one merging from the FilterTree4ML_par script
run: python FilterTree4ML_mergeAll.py cfgFileName.yml
'''

import os
import sys
import argparse
import numpy as np
import pandas as pd
import yaml
import multiprocessing as mp
from hipe4ml.tree_handler import TreeHandler

sys.path.append('..')
from utils.DfUtils import FilterBitDf, LoadDfFromRootOrParquet, GetMind0 #pylint: disable=wrong-import-position,import-error

# common bits
bitSignal = 0
bitBkg = 1
bitPrompt = 2
bitFD = 3
bitRefl = 4
# channel specific bits
# Ds
bitSecPeakDs = 9
# LctopK0s
bitLctopK0s = 9
# LctopiL
bitLctopiL = 10
# LctopKpi
bitLcNonRes = 9
bitLcLambda1520 = 10
bitLcKStar = 11
bitLcDelta = 12

def main():

    #load config file
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('configfile', metavar='text', default='cfgFileName.yml',
                        help='input config yaml file name')

    args = parser.parse_args()
    print('Opening input file')
    with open(args.configfile, 'r') as ymlCfgFile:
        cfg = yaml.load(ymlCfgFile, yaml.FullLoader)

    #Check channel
    channel = cfg['channel']
    if channel not in ['Ds', 'D0', 'Dplus', 'Dstar', 'LctopKpi', 'LctopK0s', 'LctopiL']:
        print('Error: only Ds, D0, Dplus, Dstar, LctopKpi, LctopK0s, and LctopiL channels are implemented! Exit')
        sys.exit()

    #Get filename. This parallel script only works in case 'foldername' is also provided
    inFileNames = cfg['infile']['filename']
    inFileNamesPar = cfg['infile'].get('foldername', None)
    if inFileNamesPar is not None:
        inFileNames = []
        for dirpath, dirnames, filenames in os.walk(inFileNamesPar):
            for filename in [f for f in filenames if f.endswith(".root")]:
                inFileNames.append(os.path.join(dirpath, filename))
        #print("Check if correct folders are loaded: ")
        #print(inFileNames)
    else:
        print('Error: Please run the not-parallel script or provide the correct foldername! Exit')
        sys.exit()

    #load some settings from config file
    isMC = cfg['infile']['isMC']
    outSuffix = cfg['outfile']['suffix']
    outDirName = cfg['outfile']['dirpath']

    colsToKeep = cfg['skimming']['colstokeep']
    if colsToKeep and 'inv_mass' not in colsToKeep:
        print('Warning: invariant mass branch (inv_mass) disabled. Are you sure you don\'t want to keep it?')
    if colsToKeep and 'pt_cand' not in colsToKeep:
        print('Warning: pt branch (pt_cand) disabled. Are you sure you don\'t want to keep it?')

    PtMin = cfg['skimming']['pt']['min']
    PtMax = cfg['skimming']['pt']['max']

    #make dict for bits
    bitsForSel, labelsContr = ({} for _ in range(2))
    labelsContr = {'bkg': 'Bkg', 'prompt_sig': 'Prompt', 'FD_sig': 'FD',
                    'prompt_sig_refl': 'PromptRefl', 'FD_sig_refl': 'FDRefl',
                   'prompt_sec_peak': 'SecPeakPrompt', 'FD_sec_peak': 'SecPeakFD',
                   'prompt_sig_nonreso': 'PromptNonRes', 'FD_sig_nonreso': 'FDNonRes',
                   'prompt_sig_Lambda1520': 'PromptLambda1520', 'FD_sig_Lambda1520': 'FDLambda1520',
                   'prompt_sig_KStar': 'PromptKStar', 'FD_sig_KStar': 'FDKStar',
                   'prompt_sig_Delta': 'PromptDelta', 'FD_sig_Delta': 'FDDelta'}

    if 'Dstar' in channel:
        bitsForSel = {'bkg': [bitBkg], 'prompt_sig': [bitSignal, bitPrompt], 'FD_sig': [bitSignal, bitFD]}
    elif 'Ds' in channel:
        bitsForSel = {'bkg': [bitBkg],
                      'prompt_sig': [bitSignal, bitPrompt], 'FD_sig': [bitSignal, bitFD],
                      'prompt_sig_refl': [bitSignal, bitPrompt, bitRefl], 'FD_sig_refl': [bitSignal, bitFD, bitRefl],
                      'prompt_sec_peak': [bitSecPeakDs, bitPrompt], 'FD_sec_peak': [bitSecPeakDs, bitFD]}
    elif 'Dplus' in channel:
        bitsForSel = {'bkg': [bitBkg], 'prompt_sig': [bitSignal, bitPrompt], 'FD_sig': [bitSignal, bitFD]}
    elif 'D0' in channel:
        bitsForSel = {'bkg': [bitBkg], 'prompt_sig': [bitSignal, bitPrompt], 'FD_sig': [bitSignal, bitFD],
                      'prompt_sig_refl': [bitSignal, bitPrompt, bitRefl], 'FD_sig_refl': [bitSignal, bitFD, bitRefl]}
    elif 'LctopKpi' in channel:
        bitsForSel = {'bkg': [bitBkg],
                      'prompt_sig_nonreso': [bitSignal, bitLcNonRes, bitPrompt],
                      'FD_sig_nonreso': [bitSignal, bitLcNonRes, bitFD],
                      'prompt_sig_Lambda1520': [bitSignal, bitLcLambda1520, bitPrompt],
                      'FD_sig_Lambda1520': [bitSignal, bitLcLambda1520, bitFD],
                      'prompt_sig_KStar': [bitSignal, bitLcKStar, bitPrompt],
                      'FD_sig_KStar': [bitSignal, bitLcKStar, bitFD],
                      'prompt_sig_Delta': [bitSignal, bitLcDelta, bitPrompt],
                      'FD_sig_Delta': [bitSignal, bitLcDelta, bitFD],
                      'prompt_sig_refl': [bitSignal, bitPrompt, bitRefl], 'FD_sig': [bitSignal, bitFD, bitRefl]}
    elif 'LctopK0s' in channel:
        bitsForSel = {'bkg': [bitBkg],
                      'prompt_sig': [bitSignal, bitLctopK0s, bitPrompt], 'FD_sig': [bitSignal, bitLctopK0s, bitFD]}
    elif 'LctopiL' in channel:
        bitsForSel = {'bkg': [bitBkg],
                      'prompt_sig': [bitSignal, bitLctopiL, bitPrompt], 'FD_sig': [bitSignal, bitLctopiL, bitFD]}

    #Run dataframe extraction in parallel
    paths_for_merge = []
    for inFileName in inFileNames:
        paths_for_merge.append(load_and_process_parallel(inFileName, cfg, colsToKeep, PtMin, PtMax, bitsForSel, labelsContr, bitRefl))

    #Merge the output of the parallel tasks
    print("Starting merging the dataframes!")
    if isMC:
        for idf, contr in enumerate(bitsForSel):
            paths_for_merge_idf = [paths_for_merge[ifile][idf] for ifile in range(len(paths_for_merge))]
            print(paths_for_merge_idf)

            df_mc_arr = []
            for path_merge in paths_for_merge_idf:
                if path_merge:
                    tree_handl = TreeHandler(path_merge, None)
                    df_mc_arr.append(tree_handl.get_data_frame())
                    del tree_handl

            if df_mc_arr:
                df_mc_merge = pd.concat(df_mc_arr)
                if not df_mc_merge.empty:
                    outpath_merge = f'{outDirName}/{labelsContr[contr]}{outSuffix}_pT_{PtMin:.0f}_{PtMax:.0f}.parquet.gzip'
                    print(f'Saving {labelsContr[contr]} parquet in ', outpath_merge)
                    df_mc_merge[colsToKeep].to_parquet(outpath_merge, compression='gzip')
    else:
        print(paths_for_merge)

        df_data_arr = []
        for path_merge in paths_for_merge:
            if path_merge:
                tree_handl = TreeHandler(path_merge, None)
                df_data_arr.append(tree_handl.get_data_frame())
                del tree_handl

        if df_data_arr:
            df_data_merge = pd.concat(df_data_arr)
            if not df_data_merge.empty:
                outpath_merge = f'{outDirName}/Data{outSuffix}_pT_{PtMin:.0f}_{PtMax:.0f}.parquet.gzip'
                print('Saving data to parquet in ', outpath_merge)
                df_data_merge[colsToKeep].to_parquet(outpath_merge, compression='gzip')

#######################################################################################

def load_and_process_parallel(inFileName_, cfg, colsToKeep, PtMin, PtMax, bitsForSel, labelsContr, bitRefl):

    outFolderName = os.path.dirname(os.path.abspath(inFileName_))
    inDirName = cfg['infile']['dirname']
    inTreeName = cfg['infile']['treename']
    isMC = cfg['infile']['isMC']
    outSuffix = cfg['outfile']['suffix']
    preSelections = cfg['skimming']['preselections']

    proc_paths = []
    if isMC:
        for contr in bitsForSel:
            outpath = f'{outFolderName}/{labelsContr[contr]}{outSuffix}_pT_{PtMin:.0f}_{PtMax:.0f}.parquet.gzip'
            if os.path.isfile(outpath):
                proc_paths.append(outpath)
            else:
                proc_paths.append('')
    else:
        outpath = f'{outFolderName}/Data{outSuffix}_pT_{PtMin:.0f}_{PtMax:.0f}.parquet.gzip'
        if os.path.isfile(outpath):
            proc_paths.append(outpath)
        else:
            proc_paths.append('')

    return proc_paths

#######################################################################################

if __name__ == "__main__":
    main()
