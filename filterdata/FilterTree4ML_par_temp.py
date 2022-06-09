'''
python script to filter tree from task output and save output trees in parquet files for ML studies
adjusted to run in parallel on unmerged GRID files
run: python FilterTree4ML.py cfgFileName.yml
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
    args = [(inFileName, cfg, colsToKeep, PtMin, PtMax, bitsForSel, labelsContr, bitRefl) for inFileName in inFileNames]
    paths_for_merge = multi_proc(load_and_process_parallel, args, None, 35, 30)

    #Merge the output of the parallel tasks
    print("Starting merging the dataframes!")

    print("Nevermind, don't do that now. Returning...")
    return

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

    print("Processing file: ", inFileName_)
    outFolderName = os.path.dirname(os.path.abspath(inFileName_))
    inDirName = cfg['infile']['dirname']
    inTreeName = cfg['infile']['treename']
    isMC = cfg['infile']['isMC']
    outSuffix = cfg['outfile']['suffix']
    preSelections = cfg['skimming']['preselections']

    dataFrame = LoadDfFromRootOrParquet(inFileName_, inDirName, inTreeName)

    if not colsToKeep:
        colsToKeep = list(dataFrame.columns)
        colsToKeep.remove('cand_type')

    dataFramePtCut = dataFrame.query(f'pt_cand > {PtMin} & pt_cand < {PtMax}')
    del dataFrame
    if preSelections:
        dataFramePtCutSel = dataFramePtCut.astype(float).query(preSelections)
        del dataFramePtCut
    else:
        dataFramePtCutSel = dataFramePtCut

    if cfg['infile'].get('randomly_reduce', False):
        print("Randomly reducing the number of candidates, assuming three bins...")
        dataFramePtCutSel['randFltCol'] = np.random.uniform(0.0, 1.0, len(dataFramePtCutSel))
        ptminred0 = cfg['infile']['reduce_ptbinmin'][0]
        ptmaxred0 = cfg['infile']['reduce_ptbinmax'][0]
        redfactor0 = cfg['infile']['reduce_factor'][0]
        ptminred1 = cfg['infile']['reduce_ptbinmin'][1]
        ptmaxred1 = cfg['infile']['reduce_ptbinmax'][1]
        redfactor1 = cfg['infile']['reduce_factor'][1]
        ptminred2 = cfg['infile']['reduce_ptbinmin'][2]
        ptmaxred2 = cfg['infile']['reduce_ptbinmax'][2]
        redfactor2 = cfg['infile']['reduce_factor'][2]
        dataFramePtCutSel = dataFramePtCutSel.query(f'(pt_cand > {ptminred0} & pt_cand < {ptmaxred0} & randFltCol < {redfactor0}) or (pt_cand > {ptminred1} & pt_cand < {ptmaxred1} & randFltCol < {redfactor1}) or (pt_cand > {ptminred2} & pt_cand < {ptmaxred2} & randFltCol < {redfactor2})')

    if cfg['missingvalues']['enable']:
        dataFramePtCutSel = dataFramePtCutSel.replace(cfg['missingvalues']['toreplace'], value=np.nan)

    if cfg['singletrackvars'] and cfg['singletrackvars']['addAODfiltervars']:
        # this assumes that we are analysing a 3 prong!
        if set(['pt_prong0', 'pt_prong1', 'pt_prong2']).issubset(dataFramePtCutSel.columns):
            dataFramePtCutSel['pt_prong_min'] = dataFramePtCutSel[['pt_prong0', 'pt_prong1', 'pt_prong2']].min(axis=1)
            colsToKeep.append('pt_prong_min')
            if set(['imp_par_prong0', 'imp_par_prong1', 'imp_par_prong2']).issubset(dataFramePtCutSel.columns):
                dataFramePtCutSel['imp_par_min_ptgtr2'] = dataFramePtCutSel.apply(lambda x: GetMind0(
                    [x['pt_prong0'], x['pt_prong1'], x['pt_prong2']],
                    [x['imp_par_prong0'], x['imp_par_prong1'], x['imp_par_prong2']], 2), axis=1)
                colsToKeep.append('imp_par_min_ptgtr2')

    proc_paths = []
    if isMC:
        for contr in bitsForSel:
            #print(f'Getting {labelsContr[contr]} dataframe')
            dataFramePtCutSelContr = FilterBitDf(dataFramePtCutSel, 'cand_type', bitsForSel[contr], 'and')
            # always check that it is not reflected, unless is the reflection contribution
            if 'refl' not in contr:
                dataFramePtCutSelContr = FilterBitDf(dataFramePtCutSelContr, 'cand_type', [bitRefl], 'not')

            if not dataFramePtCutSelContr.empty:
                outpath = f'{outFolderName}/{labelsContr[contr]}{outSuffix}_pT_{PtMin:.0f}_{PtMax:.0f}.parquet.gzip'
                print(f'Saving {labelsContr[contr]} parquet in ', outpath)
                dataFramePtCutSelContr[colsToKeep].to_parquet(outpath, compression='gzip')
                proc_paths.append(outpath)
            else:
                proc_paths.append('')
    else:
        outpath = f'{outFolderName}/Data{outSuffix}_pT_{PtMin:.0f}_{PtMax:.0f}.parquet.gzip'
        print('Saving data to parquet in ', outpath)
        dataFramePtCutSel[colsToKeep].to_parquet(outpath, compression='gzip')
        proc_paths.append(outpath)

    return proc_paths

#######################################################################################
#Standard functions for multiprocessing
#######################################################################################

def _callback(err):
    print(err)

def multi_proc(function, argument_list, kw_argument_list, maxperchunk, max_n_procs=10):

    chunks_args = [argument_list[x:x+maxperchunk] \
            for x in range(0, len(argument_list), maxperchunk)]
    if not kw_argument_list:
        kw_argument_list = [{} for _ in argument_list]
    chunks_kwargs = [kw_argument_list[x:x+maxperchunk] \
            for x in range(0, len(kw_argument_list), maxperchunk)]
    res_all = []
    for chunk_args, chunk_kwargs in zip(chunks_args, chunks_kwargs):
        print("Processing new chunck size=", maxperchunk)
        pool = mp.Pool(max_n_procs)
        res = [pool.apply_async(function, args=args, kwds=kwds, error_callback=_callback) \
                for args, kwds in zip(chunk_args, chunk_kwargs)]
        pool.close()
        pool.join()
        res_all.extend(res)

    res_list = None
    try:
        res_list = [r.get() for r in res_all]
    except Exception as e: # pylint: disable=broad-except
        print("EXCEPTION")
        print(e)
    return res_list

#######################################################################################

if __name__ == "__main__":
    main()
