import numpy as np
import aghast
from root_numpy import fill_hist
from ROOT import TH1D
from utils.DfUtils import MakeHist

minRange = -5.
maxRange = 5.
numBins = 100
data = np.random.normal(0, 1, int(1e7))

hroot = TH1D("hroot", "hroot", numBins, minRange, maxRange)
fill_hist(hroot, data)

nph = np.histogram(data, bins=numBins, range=(minRange, maxRange))
ghastly_h = aghast.from_numpy(nph)
hrootnew = aghast.to_root(ghastly_h, "hrootnew")
hrootnew.SetLineColor(2)

hrootutil = MakeHist(data, "hrootutil", "cc;a;b", bins=numBins, range=(minRange, maxRange))
hrootutil.SetLineColor(3)
hrootutil.Draw()
hroot.Draw("same")
hrootnew.Draw("same")

print(f'Entries: {hroot.GetEntries()} - {hrootnew.GetEntries()} - {hrootutil.GetEntries()}')
print(f'Bin Width: {hroot.GetBinWidth(1)} - {hrootnew.GetBinWidth(1)} - {hrootutil.GetBinWidth(1)}')
print(f'Bin Content: {hroot.GetBinContent(50)} - {hrootnew.GetBinContent(50)} - {hrootutil.GetBinContent(50)}')

input()
