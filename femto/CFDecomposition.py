import argparse
import yaml
import re
import os
import ctypes
import numpy as np
from ROOT import TH2D, TCanvas, TFile, gStyle, TF1, TGraphErrors, TVirtualFitter, kBlue, kOrange, TDatabasePDG, TLegend, TMath, kRed, TSpline3, kBlue
import sys
sys.path.append('..')
from utils.StyleFormatter import SetGlobalStyle, SetObjectStyle

parser = argparse.ArgumentParser(description='Arguments to pass')
parser.add_argument('cfgFileName', metavar='text', help='yaml config file name')
args = parser.parse_args()

with open(args.cfgFileName, 'r') as ymlFile:
    cfg = yaml.load(ymlFile, yaml.FullLoader)
# with open(cfg['input']['config_ptbins'], 'r') as ymlFile:
#     cfgPt = yaml.load(ymlFile, yaml.FullLoader)['cutvars']['Pt']
#     print(cfgPt)
spec = cfg['input']['specifier']
outputFileName = cfg['input']['output']
fitMethod = cfg['fit']['method']

if spec == 'PIplusDminus' or spec == 'PIplusDplus':
    suffixData = '_DPi'
elif spec == 'KplusDminus' or spec == 'KplusDplus':
    suffixData = '_DK'

# open files with CF
fSignal = TFile(os.path.join(cfg['input']['sideband_dir'], f'CFOutput_{spec}{suffixData}.root'))
fSBL = TFile(os.path.join(cfg['input']['sideband_dir'], f'CFOutput_{spec}_SBLeft.root'))
fSBR = TFile(os.path.join(cfg['input']['sideband_dir'], f'CFOutput_{spec}_SBRight.root'))
rebin = cfg['rebin']
gSignal = fSignal.Get(f'Graph_from_hCk_Reweighted_{rebin}MeV')
gSBL = fSBL.Get(f'Graph_from_hCk_Reweighted_{rebin}MeV')
gSBR = fSBR.Get(f'Graph_from_hCk_Reweighted_{rebin}MeV')
gSignal.SetTitle(';#it{k}* (MeV/c);C(#it{k}*)')
gSBL.SetTitle(';#it{k}* (MeV/c);C(#it{k}*)')
gSBR.SetTitle(';#it{k}* (MeV/c);C(#it{k}*)')

if 'legendre' in fitMethod:
    deg = int(re.findall(r'\d{1}', fitMethod)[0])
    
    fitRangeMin = cfg['fit']['range'][0]
    fitRangeMax = cfg['fit']['range'][1]

    # build series of legendre polynomials from degree 0 to l
    fitFunctionStr = ''
    for l in range(deg+1):
        if(l>0):
            fitFunctionStr += ' + '
        
        # For unclear reasons, the fit only works if fitRangeMin=0
        shift = (fitRangeMax + fitRangeMin)/2
        semiRange = (fitRangeMax - fitRangeMin)/2
        fitFunctionStr += f'[{l}]*ROOT::Math::legendre({l}, (x - {shift})/{semiRange})'
else:
    print(f'\033[41mError:\033[0m the fit mehod {fitMethod} is not implemented. Exit!')
    sys.exit()

fitFuncSBL = TF1('fitFuncSBL', fitFunctionStr, fitRangeMin, fitRangeMax)
fitFuncSBR = TF1('fitFuncSBR', fitFunctionStr, fitRangeMin, fitRangeMax)

for l in range(0, deg+1):
    fitFuncSBL.SetParLimits(l, cfg['fit']['parmin'][l], cfg['fit']['parmax'][l])
    fitFuncSBL.SetParameter(l, (cfg['fit']['parmin'][l] + cfg['fit']['parmax'][l])/2)
    fitFuncSBR.SetParLimits(l, cfg['fit']['parmin'][l], cfg['fit']['parmax'][l])
    fitFuncSBR.SetParameter(l, (0.9*cfg['fit']['parmin'][l] + cfg['fit']['parmax'][l])/2)

# Create a TGraphErrors to hold the confidence intervals
grintSBL, grintSBR = TGraphErrors(1), TGraphErrors(1)
grintSBL.SetTitle('Fit to CF SB Left with 1#sigma confidence band')
grintSBR.SetTitle('Fit to CF SB Right with 1#sigma confidence band')
nPointsCLSBL = 0
nPointsCLSBR = 0

for i in range(gSBL.GetN()):
    if fitRangeMin <= gSBL.GetX()[i] and gSBL.GetX()[i] <= fitRangeMax:
        grintSBL.SetPoint(nPointsCLSBL, gSBL.GetX()[i], 0)
        nPointsCLSBL += 1
gSBL.Fit('fitFuncSBL', 'MR+', '', fitRangeMin, fitRangeMax)
cl = int(TMath.Erf(1/TMath.Sqrt(2)))
TVirtualFitter.GetFitter().GetConfidenceIntervals(grintSBL, TMath.Erf(1./TMath.Sqrt(2))) # 1sigma conf. band


for i in range(gSBR.GetN()):
    if fitRangeMin <= gSBR.GetX()[i] and gSBR.GetX()[i] <= fitRangeMax:
        grintSBR.SetPoint(nPointsCLSBR, gSBR.GetX()[i], 0)
        nPointsCLSBR += 1
gSBR.Fit('fitFuncSBR', 'MR+', '', fitRangeMin, fitRangeMax)
TVirtualFitter.GetFitter().GetConfidenceIntervals(grintSBR, TMath.Erf(1/TMath.Sqrt(2))) # 1sigma symmetric conf. band

# Interpolate confidence bands with splines
nSBL = grintSBL.GetN()
yConfBandSBL = (ctypes.c_double * nSBL)(*[grintSBL.GetErrorYhigh(i) for i in range(nSBL)]) # Cast python list to ctypes
splineUncSBL = TSpline3('splUncCFSBL', grintSBL.GetX(), yConfBandSBL, nSBL)
nSBR = grintSBR.GetN()
yConfBandSBR = (ctypes.c_double * nSBR)(*[grintSBR.GetErrorYlow(i) for i in range(nSBL)])
splineUncSBR = TSpline3('splUncCFSBR', grintSBR.GetX(), yConfBandSBR, nSBR)

nSignal = gSignal.GetN()
# Combination of the SBL and SBR CFs
if cfg['sidebands']['combmethod'] == 'kWeighAverage':
    pass
elif cfg['sidebands']['combmethod'] == 'kAverage':
    def modelSBFunc(x, par):
        return 0.5 * (fitFuncSBL.Eval(x[0])+fitFuncSBR.Eval(x[0]))
    
    modelSB = TF1('sb', modelSBFunc, fitRangeMin, fitRangeMax)

    uncSB = [0.5 * np.sqrt(splineUncSBL.Eval(iX)**2 + splineUncSBR.Eval(iX)**2) for iX in range(nSignal)]
    splineUnc = TSpline3('splUncCF', gSignal.GetX(), (ctypes.c_double * nSBL)(*uncSB), nSignal)

else:
    print(f'\033[41mError:\033[0m the combination mehod {fitMethod} is not implemented. Exit!')
    sys.exit()

gSignalSBCorrected = TGraphErrors(nSignal)
for iP, (iX, iY) in enumerate(zip(gSignal.GetX(), gSignal.GetY())):
    lambdaParSgn = 0.7 # todo: fix fractions
    lambdaParBkg = 0.3 # todo: fix fractions
    gSignalSBCorrected.SetPoint(iP, iX, (iY - lambdaParBkg * modelSB.Eval(iX))/(1 - lambdaParBkg))

    # Gaussian error propagation
    CFRaw = gSignal.GetPointY(iP)
    CFRawUnc = gSignal.GetErrorY(iP)
    lambdaParBkgUnc = 0.01# todo: fix uncertainties
    lambdaParSgnUnc = 0.01 # todo: fix uncertainties
    CFSB = modelSB.Eval(iX)
    CFSBUnc = splineUnc.Eval(iX)

    unc2 = \
        (CFRawUnc/lambdaParSgn)**2 + \
        (CFSB*lambdaParBkgUnc/lambdaParSgn)**2 + \
        (lambdaParBkg*CFSBUnc/lambdaParSgn)**2 + \
        (CFRaw/lambdaParSgn/lambdaParSgn - lambdaParBkg*CFSB/lambdaParSgn/lambdaParSgn)**2
    gSignalSBCorrected.SetPointError(iP, gSignal.GetErrorX(iP), np.sqrt(unc2))
    



# Make plots and drawings
oFile = TFile(f'{outputFileName}.root', 'RECREATE')
gStyle.SetOptFit(11111111)
SetGlobalStyle()

# Left sideband
cSBL = TCanvas('cCFSBL', 'Correlation function SBL', 600, 600)
SetObjectStyle(gSBL)
gSBL.Draw('alpe')
fitFuncSBL.SetLineColor(kRed)
fitFuncSBL.SetMarkerColor(kRed)
grintSBL.SetLineColor(kOrange + 10)
grintSBL.SetMarkerColor(kOrange + 10)
grintSBL.SetFillColorAlpha(kOrange + 6, 0.7)
grintSBL.Draw("3 same")

legSBL = TLegend(0.45, 0.15, '')
legSBL.AddEntry(fitFuncSBL, "Leg. Pol. fit")
legSBL.AddEntry(grintSBL, "1#sigma CL")
legSBL.AddEntry(gSBL, "SB left")
legSBL.Draw()
cSBL.Print(f'{outputFileName}_fitSBL.pdf')
cSBL.Write()

# Right sideband
cSBR = TCanvas('cCFSBR', 'Correlation function SBR', 600, 600)
SetObjectStyle(gSBR)
gSBR.Draw('alpe')
fitFuncSBR.SetLineColor(kRed)
fitFuncSBR.SetMarkerColor(kRed)
grintSBR.SetLineColor(kOrange + 10)
grintSBR.SetMarkerColor(kOrange + 10)
grintSBR.SetFillColorAlpha(kOrange + 6, 0.7)
grintSBR.Draw("3 same")

legSBR = TLegend(0.45, 0.15, '')
legSBR.AddEntry(fitFuncSBR, "Leg. Pol. fit")
legSBR.AddEntry(grintSBR, "1#sigma CL")
legSBR.AddEntry(gSBR, "SB left")
legSBR.Draw()
cSBR.Print(f'{outputFileName}_fitSBR.pdf')
cSBR.Write()
################################################################################

# Comparison between before and after SB correction
cCFSBCorrected = TCanvas('cCFSBCorrected', 'with and without correction', 600, 600)
SetObjectStyle(gSignal)
SetObjectStyle(gSignalSBCorrected, markercolor=kBlue)
gSignal.Draw('ape')
gSignalSBCorrected.SetLineColor(kBlue)
legSBCorrected = TLegend(0.45, 0.15, '')
legSBCorrected.AddEntry(gSignal, 'Before SB correction')
legSBCorrected.AddEntry(gSignalSBCorrected, 'After SB correction')
legSBCorrected.Draw()
gSignalSBCorrected.Draw('pe same')




# # Computation of weights
# ryFile = TFile(cfg['input']['ry_filename'])
# nsigma = cfg['sidebands']['nsigma']
# dDecChannel = cfg['input']['ddecchannel']
# massDplus = TDatabasePDG.Instance().GetParticle(411).Mass()
# massDs = TDatabasePDG.Instance().GetParticle(431).Mass()
# massLc = TDatabasePDG.Instance().GetParticle(4122).Mass()
# massDstar = TDatabasePDG.Instance().GetParticle(413).Mass() - TDatabasePDG.Instance().GetParticle(421).Mass()
# massD0 = TDatabasePDG.Instance().GetParticle(421).Mass()

# # if particleName == 'Dplus':
# #     massForFit=massDplus
# # elif particleName == 'Ds':
# #     massForFit = massDs
# # elif particleName == 'Dstar':
# #     massForFit = massDstar
# # elif particleName == 'D0':
# #     massForFit = massD0
# # else:
# #     massForFit = massLc
# gWeightsLeft = TGraphErrors(1)
# gWeightsRight = TGraphErrors(1)
# for iPt, (ptMin, ptMax) in enumerate(zip(cfgPt['min'], cfgPt['max'])):
#     iPtMean = (ptMin + ptMax)/2
#     iPtWidth = (ptMax - ptMin)/2
#     fBkg = ryFile.Get(f'fBkg_{ptMin:.1f}_{ptMax:.1f}')
#     if dDecChannel == 'kDplustoKpipi':
#         massWidth = 0.006758 + iPtMean * 0.0005124 # Dplus2Kpipi@pp13TeV only. Taken from https://github.com/alisw/AliPhysics/blob/f5074bffd6962915a4b36842e2a5e37f60e544d2/PWGCF/FEMTOSCOPY/FemtoDream/AliAnalysisTaskCharmingFemto.cxx#L1069
#         mass = massDplus
#     wLeft = fBkg.Integral(massDplus - nsigma*massWidth, massDplus, 1e-6)
#     wRight = fBkg.Integral(massDplus, massDplus + nsigma*massWidth, 1e-6)
#     # wLeftError = fBkg.IntegralError(massDplus - nsigma*massWidth, massDplus)
#     # wRightError = fBkg.IntegralError(massDplus, massDplus + nsigma*massWidth)
#     print("vkjsnkvjnsf ", fBkg.GetParError(0))
#     wLeftNorm = wLeft/(wLeft + wRight)
#     wRightNorm = wRight/(wLeft + wRight)
#     gWeightsLeft.SetPoint(iPt, iPtMean, wLeftNorm,)
#     gWeightsRight.SetPoint(iPt, iPtMean, wRightNorm,)
#     print(wLeftNorm, wRightNorm)


# c = TCanvas("a", "left", 600, 600)
# # SetGlobalStyle(padrightmargin=0.01)
# SetObjectStyle(gWeightsLeft, markerstyle=21, color = 2)
# SetObjectStyle(gWeightsRight)
# gWeightsLeft.Draw('ape')
# gWeightsLeft.SetTitle(';p_{T} (GeV/c); w')
# gWeightsLeft.GetYaxis().SetRangeUser(0.45, 0.55)
# gWeightsRight.Draw('pe same')