'''
Script to do template fits to MC and estimate the fraction of primary/secondary/material pions and kaons
'''

import argparse
import sys
from typing import Collection
import ctypes
import numpy as np
import yaml
from ROOT import TCanvas, TH1D, TFractionFitter, TFile, TObjArray, TColor, gStyle, TDirectory, kBlue, kRed, kGreen, kMagenta, TLatex, TLegend

parser = argparse.ArgumentParser(description='Arguments to pass')
parser.add_argument('cfgFileName', metavar='text', help='yaml config file name')
args = parser.parse_args()

with open(args.cfgFileName, 'r') as ymlConfig:
    cfg = yaml.load(ymlConfig, yaml.FullLoader)

channel = cfg['input']['channel']
dataFile = TFile(cfg['input']['data'], 'read')
mcFile = TFile(cfg['input']['mc'], 'read')
sourceList = cfg['sources']
oFile = TFile(cfg['output']['filename'], 'recreate')

print(sourceList)
if channel == 'DK':
    buddy = 'kaon'
elif channel == 'Dpi':
    buddy = 'pion'
else:
    print(f'\033[41mError:\033[0m the {channel} channel is not supported. Exit!')
    sys.exit()

dirname_data = f'HM_CharmFemto_D{buddy}_TrackCuts0/HM_CharmFemto_D{buddy}_TrackCuts0'
dirname_mc = f'HM_CharmFemto_D{buddy}_TrackCutsMC0/HM_CharmFemto_D{buddy}_TrackCutsMC0'

hDcaPtList_data = dataFile.Get(dirname_data)
hDcaPtList_mc = mcFile.Get(dirname_mc)

hDcaPt_data = hDcaPtList_data.FindObject('DCAXYPtBinningTot')
hDcaPt_data.RebinX(10)
hDcaPt_data.GetYaxis().SetRangeUser(-0.2, 0.2)


cList = []
for iPtBin in range(0, 1):
    hDataToFit = hDcaPt_data.ProjectionY(f'hDca_{iPtBin+1}_data', iPtBin+1, iPtBin+1)
    canv = TCanvas(f'c_pT{iPtBin+1}_data', f'c_pT{iPtBin+1}_data', 600, 600)
    cList.append(canv)
    hDataToFit.Draw("hist")
    # hDataToFit.Rebin(5)
    hDataToFit.Scale(1./hDataToFit.GetEntries())

    hMcToFit = TObjArray(len(sourceList))
    
    for iSource, source in enumerate(sourceList):
        hDcaPt_mc = hDcaPtList_mc.FindObject('DCAPtBinning').FindObject(f'DCAPtBinning{source}')
        # if hDcaPt_mc.GetEntries() < 1000:
        #     continue
        hDcaPt_mc.RebinX(10)
        hDcaPt_mc.GetYaxis().SetRangeUser(-0.2, 0.2)

        if hDcaPt_data.GetNbinsX() != hDcaPt_mc.GetNbinsX():
            print('\033[31mError\033[0m: number of pT bins are not the same for data and MC. Exit!')
            sys.exit()
        if hDcaPt_data.GetXaxis().GetBinLowEdge(iPtBin+1) - hDcaPt_mc.GetXaxis().GetBinLowEdge(iPtBin+1) > 1e-6:
            print('\033[31mError\033[0m: low edge of the pT bins are not the same for data and MC. Exit!')
            sys.exit()
        hMcToFit.Add(hDcaPt_mc.ProjectionY(f'hDca_{source}_{iPtBin+1}_mc', iPtBin+1, iPtBin+1))
        c = TCanvas(f'c_pT{iPtBin+1}_{source}', f'c_pT{iPtBin+1}_{source}', 600, 600)
        cList.append(c)

        hMcToFit.At(iSource).SetMarkerStyle(20+iSource)
        hMcToFit.At(iSource).SetMarkerSize(1)
        # hMcToFit.At(iSource).Rebin(2)
        # cList.append(TCanvas(f'c_pT{iPtBin+1}_{source}', f'c_pT{iPtBin+1}_{source}', 600, 600))
        hMcToFit.At(iSource).SetLineColor(kBlue+iSource)
        hMcToFit.At(iSource).SetMarkerColor(kBlue+iSource)
        hMcToFit.At(iSource).Draw('hist')

        hMcToFit.At(iSource).Write()
        if hMcToFit[iSource].GetEntries() < 0:
            hMcToFit[iSource].Rem
        hMcToFit[iSource].Scale(1./hMcToFit[iSource].GetEntries())
    # break
    
    #ratio tot/pri
    # hRatio = hDataToFit.Clone('hRatio')
    # hRatio.Scale(1./hRatio.GetEntries())
    # hRatio.Divide(hMcToFit[0])
    # cR = TCanvas('CR', 'cR')
    # hRatio.Draw("hist")

    # # input()
    cocktail = TFractionFitter(hDataToFit, hMcToFit)
    print(hMcToFit.At(0).GetEntries())
    print(hMcToFit.At(1).GetEntries())
    print(hMcToFit.At(2).GetEntries())
    s = hMcToFit.At(0).GetEntries()+hMcToFit.At(1).GetEntries()+hMcToFit.At(2).GetEntries()
    print(hMcToFit.At(0).GetEntries()/s)
    print(hMcToFit.At(1).GetEntries()/s)
    print(hMcToFit.At(2).GetEntries()/s)
    
    for iSource in range(len(sourceList)):
        fractionInMC = hMcToFit.At(iSource).GetEntries()/s
        cocktail.Constrain(iSource, 0.5*fractionInMC, min(1, 2*fractionInMC))
    cFit = TCanvas(f'c_pT{iPtBin+1}_Fit', f'c_pT{iPtBin+1}_Fit', 600, 600)
    status = cocktail.Fit()
    print('chi2: ', cocktail.GetChisquare(), 'NDF: ', cocktail.GetNDF())
    hDataToFit.SetMarkerStyle(20)
    hDataToFit.SetMarkerSize(0.5)
    hDataToFit.SetMarkerColor(kRed)
    hDataToFit.SetLineColor(kRed)
    hDataToFit.Draw("ep")
    
    hSum = TH1D('hSum', 'Sum', hMcToFit.At(0).GetNbinsX(), -0.2, 0.2)
    hSum.Sumw2()
    frac = [ctypes.c_double() for iSource in range(len(sourceList))]
    unc = [ctypes.c_double() for iSource in range(len(sourceList))]

    leg = TLegend()
    leg.AddEntry(hDataToFit, 'data')
    for iSource, source in enumerate(sourceList):
        cocktail.GetResult(iSource, frac[iSource], unc[iSource])
        hMcToFit.At(iSource).Scale(frac[iSource].value)
        hMcToFit.At(iSource).Draw('same')
        hSum.Add(hMcToFit.At(iSource))
        leg.AddEntry(hMcToFit.At(iSource), sourceList[iSource])
    # leg.Draw('same')
    # result = cocktail.GetPlot()
    # result.SetLineColor(kGreen)
    # result.Draw('same')
    hSum.SetLineColor(kMagenta)
    hSum.SetMarkerColor(kMagenta)
    hSum.Draw('hist same')
    Tl = TLatex()
    tot = 0
    for iSource in range(len(sourceList)):
        tot += float(frac[iSource].value)
    print('Entries in MC hist')
    print(sourceList[0], ': ', hMcToFit.At(0).GetEntries())
    print(sourceList[1], ': ', hMcToFit.At(1).GetEntries())
    print(sourceList[2], ': ', hMcToFit.At(2).GetEntries())
    s = hMcToFit.At(0).GetEntries()+hMcToFit.At(1).GetEntries()+hMcToFit.At(2).GetEntries()
    print('Fractions in MC')
    print(sourceList[0], ': ', hMcToFit.At(0).GetEntries()/s)
    print(sourceList[1], ': ', hMcToFit.At(1).GetEntries()/s)
    print(sourceList[2], ': ', hMcToFit.At(2).GetEntries()/s)
    print('sum frac form TFractionFitter: ', tot)
    # Tl.SetTextAlign(12)
    Tl.SetTextSize(0.4)
    Tl.DrawLatex(0.1,0.5, f'#chi^{{2}} = {cocktail.GetChisquare()}; NDF = {cocktail.GetNDF()}')
    # Tl.Draw('same')
    # break
# oFile.mkdir(f'HM_CharmFemto_D{buddy}_TrackCuts0')
# oFile.cd(f'HM_CharmFemto_D{buddy}_TrackCuts0')
# hDcaPtList_data.Write()
# # dataFile.Get(f'HM_CharmFemto_D{buddy}_TrackCuts0').Write()
# oFile.mkdir(f'HM_CharmFemto_D{buddy}_TrackCutsMC0')
# oFile.cd(f'HM_CharmFemto_D{buddy}_TrackCutsMC0')
# mcFile.Get(f'HM_CharmFemto_D{buddy}_TrackCutsMC0').Write()
# hDcaPtList_mc.Write()

# oFile.Close()