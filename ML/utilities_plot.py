from array import array
import math
import numpy as np
from root_numpy import fill_hist
from ROOT import TH1F, TH2F, TH3F

def buildhisto(h_name, h_tit, arrayx, arrayy=None, arrayz=None):
    """
    Create a histogram of size 1D, 2D, 3D, depending on the number of arguments given
    """
    histo = None
    def binning(binning_array):
        return len(binning_array) - 1, binning_array
    if arrayz:
        histo = TH3F(h_name, h_tit, *binning(arrayx), *binning(arrayy), *binning(arrayz))
    elif arrayy:
        histo = TH2F(h_name, h_tit, *binning(arrayx), *binning(arrayy))
    else:
        histo = TH1F(h_name, h_tit, *binning(arrayx))
    histo.Sumw2()
    return histo

def build2dhisto(titlehist, arrayx, arrayy):
    """
    Create a TH2 histogram from two axis arrays.
    """
    return buildhisto(titlehist, titlehist, arrayx, arrayy)

def makefill2dhist(df_, titlehist, arrayx, arrayy, nvar1, nvar2):
    """
    Create a TH2F histogram and fill it with two variables from a dataframe.
    """
    histo = build2dhisto(titlehist, arrayx, arrayy)
    df_rd = df_[[nvar1, nvar2]]
    arr2 = df_rd.to_numpy()
    fill_hist(histo, arr2)
    return histo

def buildbinning(nbinsx, xlow, xup):
    """
    Build a list for binning out of bin limits and number of bins
    """
    listnumber = [xlow + (xup - xlow) / nbinsx * i for i in range(nbinsx + 1)]
    return buildarray(listnumber)

def buildarray(listnumber):
    """
    Build an array out of a list, useful for histogram binning
    """
    arraynumber = array('d', listnumber)
    return arraynumber
