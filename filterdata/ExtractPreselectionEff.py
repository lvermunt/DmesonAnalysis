'''
python script to extract the preselection efficiency
run: python ExtractPreselectionEff.py cfgFileName.yml
'''

import os
import sys
import argparse
from array import *
import multiprocessing as mp
import yaml
from ROOT import TFile, TH1F

sys.path.append('..')
from utils.DfUtils import FilterBitDf, LoadDfFromRootOrParquet #pylint: disable=wrong-import-position,import-error

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
    if not isMC:
        print('Error: Efficiency can only be extracted for MC! Exit.')
        sys.exit()

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
    args = [(inFileName, cfg, bitsForSel, labelsContr, bitRefl) for inFileName in inFileNames]
    paths_for_merge = multi_proc(load_and_process_parallel, args, None, 35, 30)


#######################################################################################

def load_and_process_parallel(inFileName_, cfg, bitsForSel, labelsContr, bitRefl):

    print("Processing file: ", inFileName_)
    outFolderName = os.path.dirname(os.path.abspath(inFileName_))
    inDirName = cfg['infile']['dirname']
    inTreeName = cfg['infile']['treename']
    isMC = cfg['infile']['isMC']
    outSuffix = cfg['outfile']['suffix']
    preSelections = cfg['skimming']['preselections']

    ptranges = cfg['skimming']['analysisptbins']
    nptbins = len(ptranges) - 1

    dataFrame = LoadDfFromRootOrParquet(inFileName_, inDirName, inTreeName)

    if preSelections:
        dataFrameSel = dataFrame.astype(float).query(preSelections)
        del dataFrame
    else:
        dataFrameSel = dataFrame

    sparsePrompt, sparseFD = LoadSparse(inFileName_, cfg)

    proc_paths = []
    if isMC:
        for contr in bitsForSel:
            dataFrameSelContr = FilterBitDf(dataFrameSel, 'cand_type', bitsForSel[contr], 'and')
            # always check that it is not reflected, unless is the reflection contribution
            if 'refl' not in contr:
                dataFrameSelContr = FilterBitDf(dataFrameSelContr, 'cand_type', [bitRefl], 'not')

            if not dataFrameSelContr.empty:
                outpath = f'{outFolderName}/{labelsContr[contr]}{outSuffix}_preSelEfficiency.root'
                print(f'Saving {labelsContr[contr]} root in ', outpath)

                hpt_rec = TH1F("hpt_rec", ";#it{p}_{T} (GeV/#it{c});Counts", nptbins, array("d", ptranges))
                for pt in dataFrameSelContr['pt_cand'].to_numpy():
                    hpt_rec.Fill(pt)

                hpt_gen = TH1F("hpt_gen", ";#it{p}_{T} (GeV/#it{c});Counts", nptbins, array("d", ptranges))
                print(labelsContr[contr])
                if labelsContr[contr] == 'Prompt':
                    htemp = sparsePrompt.Projection(0)
                    print(htemp.GetBinContent(2), htemp.GetBinCenter(2))
                    htemp = htemp.Rebin(nptbins, "htemp_reb", array("d", ptranges))
                    print(htemp.GetBinContent(2), htemp.GetBinCenter(2))
                    for ipt in range(nptbins+1):
                        hpt_gen.SetBinContent(ipt, htemp.GetBinContent(ipt))
                        hpt_gen.SetBinError(ipt, htemp.GetBinError(ipt))
                if labelsContr[contr] == 'FD':
                    htemp = sparseFD.Projection(0)
                    htemp = htemp.Rebin(nptbins, "htemp_reb", array("d", ptranges))
                    for ipt in range(nptbins+1):
                        hpt_gen.SetBinContent(ipt, htemp.GetBinContent(ipt))
                        hpt_gen.SetBinError(ipt, htemp.GetBinError(ipt))

                outfile = TFile.Open(outpath, 'RECREATE')
                outfile.cd()
                hpt_rec.Write()
                hpt_gen.Write()
                outfile.Close()
                proc_paths.append(outpath)
            else:
                proc_paths.append('')

    return proc_paths

def LoadSparse(inFileName_, cfg):

    inDirName = cfg['infile']['dirname']
    inListName = cfg['infile']['listname']
    inSparseNamePrompt = cfg['infile']['sparsenameprompt']
    inSparseNameFD = cfg['infile']['sparsenamefd']

    infileData = TFile(inFileName_)
    indirData = infileData.Get(inDirName)
    if not indirData:
        print(f'Directory {inDirName} not found!')
        return None, None
    inlistData = indirData.Get(inListName)
    if not inlistData:
        print(f'List {inListName} not found!')
        return None, None

    sparsesGenPrompt = inlistData.FindObject(inSparseNamePrompt)
    if not sparsesGenPrompt:
        print(f'ERROR: sparse {inSparseNamePrompt} not found!')
        return None, None
    sparsesGenFD = inlistData.FindObject(inSparseNameFD)
    if not sparsesGenFD:
        print(f'ERROR: sparse {inSparseNameFD} not found!')
        return None, None
    infileData.Close()

    return sparsesGenPrompt, sparsesGenFD

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
