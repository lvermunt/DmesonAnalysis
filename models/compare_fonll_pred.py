"""
Utility script to compare the FONLL predictions for B and D from B
"""
import sys
import pandas as pd
import numpy as np
from ROOT import TCanvas, TLegend, gPad #pylint: disable=import-error,no-name-in-module
from ROOT import kRed, kAzure, kFullCircle, kFullSquare #pylint: disable=import-error,no-name-in-module
sys.path.append('..')
from utils.StyleFormatter import SetGlobalStyle, SetObjectStyle #pylint: disable=wrong-import-position,import-error
from utils.DfUtils import MakeHist #pylint: disable=wrong-import-position,import-error

def main(): #pylint: disable=too-many-statements
    """
    main function of script
    """
    SetGlobalStyle(padleftmargin=0.18, padbottommargin=0.14, titleoffsety=1.6, maxdigits=2, optstat=0)

    n_bins = 1001
    bin_width = 0.05 # GeV
    bin_lims = [i * bin_width for i in range(0, n_bins + 1)]
    fonllBtoD_df = pd.read_csv("fonll/FONLL_DfromB_pp5_y05_toCook.txt", sep=" ", header=14).astype('float64')
    fonllB_df = pd.read_csv("fonll/FONLL_B_pp5_y05.txt", sep=" ", header=13).astype('float64')

    hFONLLBtoDCentral = MakeHist(fonllBtoD_df['pt'].to_numpy(), 'hDfromBpred_central',
                                 ';#it{p}_{T} (GeV/#it{c});#frac{d#sigma}{d#it{p}_{T}} (pb GeV^{-1} #it{c})',
                                 bins=bin_lims, weights=fonllBtoD_df['central'].to_numpy())
    hFONLLBtoDMin = MakeHist(fonllBtoD_df['pt'].to_numpy(), 'hDfromBpred_min',
                             ';#it{p}_{T} (GeV/#it{c});#frac{d#sigma}{d#it{p}_{T}} (pb GeV^{-1} #it{c})',
                             bins=bin_lims, weights=fonllBtoD_df['min'].to_numpy())
    hFONLLBtoDMax = MakeHist(fonllBtoD_df['pt'].to_numpy(), 'hDfromBpred_max',
                             ';#it{p}_{T} (GeV/#it{c});#frac{d#sigma}{d#it{p}_{T}} (pb GeV^{-1} #it{c})',
                             bins=bin_lims, weights=fonllBtoD_df['max'].to_numpy())
    hFONLLBCentral = MakeHist(fonllB_df['pt'].to_numpy(), 'hBpred_central',
                              ';#it{p}_{T} (GeV/#it{c});#frac{d#sigma}{d#it{p}_{T}} (pb GeV^{-1} #it{c})',
                              bins=bin_lims, weights=fonllB_df['central'].to_numpy())
    hFONLLBMin = MakeHist(fonllB_df['pt'].to_numpy(), 'hBpred_min',
                          ';#it{p}_{T} (GeV/#it{c});#frac{d#sigma}{d#it{p}_{T}} (pb GeV^{-1} #it{c})',
                          bins=bin_lims, weights=fonllB_df['min'].to_numpy())
    hFONLLBMax = MakeHist(fonllB_df['pt'].to_numpy(), 'hBpred_max',
                          ';#it{p}_{T} (GeV/#it{c});#frac{d#sigma}{d#it{p}_{T}} (pb GeV^{-1} #it{c})',
                          bins=bin_lims, weights=fonllB_df['max'].to_numpy())

    for iBin in range(hFONLLBtoDCentral.GetNbinsX()):
        hFONLLBtoDCentral.SetBinError(iBin + 1, 1.e-3)
        hFONLLBtoDMin.SetBinError(iBin + 1, 1.e-3)
        hFONLLBtoDMax.SetBinError(iBin + 1, 1.e-3)
        hFONLLBCentral.SetBinError(iBin + 1, 1.e-3)
        hFONLLBMin.SetBinError(iBin + 1, 1.e-3)
        hFONLLBMax.SetBinError(iBin + 1, 1.e-3)


    hFONLL = [hFONLLBtoDCentral, hFONLLBtoDMin, hFONLLBtoDMax]
    hBFONLL = [hFONLLBCentral, hFONLLBMin, hFONLLBMax]
    labels = ['Central', 'Min', 'Max']
    canvas = []

    for label in labels:
        canvas.append(TCanvas(f'cComp{label}', '', 2000, 1000))

    for (histo, histo_std, label, canv) in zip(hFONLL, hBFONLL, labels, canvas):
        pt_lims = np.array(histo.GetXaxis().GetXbins(), 'd')

        pt_min = list(pt_lims)[0]
        pt_max = list(pt_lims)[-1]
        histo_std = histo_std.Rebin(histo.GetNbinsX(), f'hBFONLL{label}', pt_lims)
        histo.Rebin(5)
        histo_std.Rebin(5)
        histo.Scale(1.e-6)
        histo_std.Scale(1.e-6)
        histo_ratio = histo.Clone(f'hRatioDfromBOverB{label}')
        histo_ratio.Divide(histo, histo_std)
        histo_ratio.GetYaxis().SetTitle('FONLL D from B / FONLL B')

        print(f'\nFONLL B {label} integral (ub): {histo_std.Integral("width"):.4e}')
        print(f'FONLL B from D {label} integral (ub): {histo.Integral("width"):.4e}\n')

        canv.Divide(2, 1)
        canv.cd(1).DrawFrame(pt_min, 1e-4, pt_max, 100.,
                             ';#it{p}_{T} (GeV/#it{c});#frac{d#sigma}{d#it{p}_{T}} (#mub GeV^{-1} #it{c})')
        leg = TLegend(0.4, 0.7, 0.8, 0.9)
        leg.SetTextSize(0.045)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        SetObjectStyle(histo, color=kAzure+4, markerstyle=kFullSquare)
        SetObjectStyle(histo_std, color=kRed, markerstyle=kFullCircle)
        histo.Draw('same')
        histo_std.Draw('same')
        leg.AddEntry(histo, f'FONLL D from B {label}', 'p')
        leg.AddEntry(histo_std, f'FONLL B {label}', 'p')
        gPad.SetLogy()
        leg.Draw()

        hframe_ratio = canv.cd(2).DrawFrame(pt_min, 0., pt_max, 3.5,
                                            ';#it{p}_{T} (GeV/#it{c}); FONLL D from B / FONLL B')
        hframe_ratio.GetYaxis().SetDecimals()
        SetObjectStyle(histo_ratio, color=kAzure+4, markerstyle=kFullSquare)
        histo_ratio.Draw('same')
        canv.Update()
        canv.SaveAs(f'CompFONLL_DfromBvsB_{label}.pdf')

    input('Press enter to exit')

main()
