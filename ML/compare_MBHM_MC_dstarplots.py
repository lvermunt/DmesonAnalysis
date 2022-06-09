'''
python script to check variable distribution using the hipe4ml package
run: python ChecckDistr.py cfgFileName.yml cutSetFileName.yml
'''
import sys
import argparse
from dedicated_to_root_plotting import extract_topo_distr_dstar

sys.path.append('..')
from utils.DfUtils import LoadDfFromRootOrParquet #pylint: disable=wrong-import-position,import-error,no-name-in-module

child_list = ["child_1", "child_2", "child_3"]

for child in child_list:
    dfHM = LoadDfFromRootOrParquet(f'/data/TTree/DstarFemto/vAN-20211121_ROOT6-1/pp_cern.ch/654_20211125-1053/unmerged/{child}/Prompt__Dstar_MC_20161718_HM_Pola_vDistrTest_pT_0_50.parquet.gzip', None, None)
    dfMB = LoadDfFromRootOrParquet(f'/data/TTree/DstarFemto/vAN-20211121_ROOT6-1/pp_cern.ch/656_20211125-1054/unmerged/{child}/Prompt__Dstar_MC_20161718_HM_Pola_vDistrTest_pT_0_50.parquet.gzip', None, None)

    ptmin = [1, 2, 4, 6, 8,  12]
    ptmax = [2, 4, 6, 8, 12, 50]

    iPt = 0
    for (ptMin, ptMax) in zip(ptmin, ptmax):
        print(f'Projecting distributions for {ptMin:.1f} < pT < {ptMax:.1f} GeV/c')
        dfHMCut = dfHM.query(f'{ptMin} < pt_cand < {ptMax}')
        dfMBCut = dfMB.query(f'{ptMin} < pt_cand < {ptMax}')
        topovars = ['d_len', 'd_len_xy', 'norm_dl_xy', 'cos_p', 'cos_p_xy', 'dca', 'imp_par_xy', 'max_norm_d0d0exp', 'imp_par_prod', 'nsigComb_Pi_1', 'nsigComb_K_1', 'nsigComb_Pi_2', 'nsigComb_K_2']
        extract_topo_distr_dstar([ptMin, ptMax], f'/data/TTree/DstarFemto/vAN-20211121_ROOT6-1/pp_cern.ch/654_20211125-1053/unmerged/{child}/', topovars, dfHMCut, dfMBCut, None)
