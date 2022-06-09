from ROOT import TFile, TH1F
from root_numpy import fill_hist

def extract_topo_distr_dstar(PtBin, OutPutDirPt, vars_to_plot, PromptDf, FDDf, BkgDf): #pylint: disable=too-many-statements, too-many-branches
    '''
    function to plot the topological variables of the Dstar in a nice way
    '''

    out_file = TFile.Open(f'{OutPutDirPt}/topological_vars_Dstar_{PtBin[0]}_{PtBin[1]}.root', 'recreate')

    nbins = 160
    hbkg_d_len = TH1F("hbkg_d_len", ";#it{L} (#mum);Normalised counts", nbins, 0, 5000)
    hpr_d_len = TH1F("hpr_d_len", ";#it{L} (#mum);Normalised counts", nbins, 0, 5000)
    hfd_d_len = TH1F("hfd_d_len", ";#it{L} (#mum);Normalised counts", nbins, 0, 5000)
    hbkg_d_len_xy = TH1F("hbkg_d_len_xy", ";#it{L}_{xy} (#mum);Normalised counts", nbins, 0, 5000)
    hpr_d_len_xy = TH1F("hpr_d_len_xy", ";#it{L}_{xy} (#mum);Normalised counts", nbins, 0, 5000)
    hfd_d_len_xy = TH1F("hfd_d_len_xy", ";#it{L}_{xy} (#mum);Normalised counts", nbins, 0, 5000)
    hbkg_norm_dl_xy = TH1F("hbkg_norm_dl_xy", ";#it{L}_{xy} / #sigma_{}(#it{L}_{xy});Normalised counts", nbins, 0, 30)
    hpr_norm_dl_xy = TH1F("hpr_norm_dl_xy", ";#it{L}_{xy} / #sigma_{}(#it{L}_{xy});Normalised counts", nbins, 0, 30)
    hfd_norm_dl_xy = TH1F("hfd_norm_dl_xy", ";#it{L}_{xy} / #sigma_{}(#it{L}_{xy});Normalised counts", nbins, 0, 30)
    hbkg_cos_p = TH1F("hbkg_cos_p", ";cos(#theta_{p});Normalised counts", 5*nbins, -1, 1)
    hpr_cos_p = TH1F("hpr_cos_p", ";cos(#theta_{p});Normalised counts", 5*nbins, -1, 1)
    hfd_cos_p = TH1F("hfd_cos_p", ";cos(#theta_{p});Normalised counts", 5*nbins, -1, 1)
    hbkg_cos_p_xy = TH1F("hbkg_cos_p_xy", ";cos(#theta_{p}^{xy});Normalised counts", 5*nbins, -1, 1)
    hpr_cos_p_xy = TH1F("hpr_cos_p_xy", ";cos(#theta_{p}^{xy});Normalised counts", 5*nbins, -1, 1)
    hfd_cos_p_xy = TH1F("hfd_cos_p_xy", ";cos(#theta_{p}^{xy});Normalised counts", 5*nbins, -1, 1)
    hbkg_imp_par_xy = TH1F("hbkg_imp_par_xy", ";|#it{d}_{0}^{#kern[0.1]{xy}}| (#mum);Normalised counts", nbins, 0, 300)
    hpr_imp_par_xy = TH1F("hpr_imp_par_xy", ";|#it{d}_{0}^{#kern[0.1]{xy}}| (#mum);Normalised counts", nbins, 0, 300)
    hfd_imp_par_xy = TH1F("hfd_imp_par_xy", ";|#it{d}_{0}^{#kern[0.1]{xy}}| (#mum);Normalised counts", nbins, 0, 300)
    hbkg_delta_mass_D0 = TH1F("hbkg_delta_mass_D0", ";|#Delta#it{M}(D^{0})| (MeV/#it{c}^{2});Normalised counts", nbins, 0, 100)
    hpr_delta_mass_D0 = TH1F("hpr_delta_mass_D0", ";|#Delta#it{M}(D^{0})| (MeV/#it{c}^{2});Normalised counts", nbins, 0, 100)
    hfd_delta_mass_D0 = TH1F("hfd_delta_mass_D0", ";|#Delta#it{M}(D^{0})| (MeV/#it{c}^{2});Normalised counts", nbins, 0, 100)
    hbkg_max_norm_d0d0exp = TH1F("hbkg_max_norm_d0d0exp", ";|#it{f}(#it{d}_{0}^{#kern[0.1]{xy}}, #sigma_{}(#it{d}_{0}^{#kern[0.1]{xy}}))|;Normalised counts", nbins, 0, 20)
    hpr_max_norm_d0d0exp = TH1F("hpr_max_norm_d0d0exp", ";|#it{f}(#it{d}_{0}^{#kern[0.1]{xy}}, #sigma_{}(#it{d}_{0}^{#kern[0.1]{xy}}))|;Normalised counts", nbins, 0, 20)
    hfd_max_norm_d0d0exp = TH1F("hfd_max_norm_d0d0exp", ";|#it{f}(#it{d}_{0}^{#kern[0.1]{xy}}, #sigma_{}(#it{d}_{0}^{#kern[0.1]{xy}}))|;Normalised counts", nbins, 0, 20)
    hbkg_imp_par_prod = TH1F("hbkg_imp_par_prod", ";#it{d}_{0,#it{i}}^{#kern[0.1]{xy}}#times#it{d}_{0,#it{j}}^{#kern[0.1]{xy}} (#mum^{2});Normalised counts", nbins, -500000, 500000)
    hpr_imp_par_prod = TH1F("hpr_imp_par_prod", ";#it{d}_{0,#it{i}}^{#kern[0.1]{xy}}#times#it{d}_{0,#it{j}}^{#kern[0.1]{xy}} (#mum^{2});Normalised counts", nbins, -500000, 500000)
    hfd_imp_par_prod = TH1F("hfd_imp_par_prod", ";#it{d}_{0,#it{i}}^{#kern[0.1]{xy}}#times#it{d}_{0,#it{j}}^{#kern[0.1]{xy}} (#mum^{2});Normalised counts", nbins, -500000, 500000)
    hbkg_cos_t_star = TH1F("hbkg_cos_t_star", ";cos(#theta*);Normalised counts", nbins, -1, 1)
    hpr_cos_t_star = TH1F("hpr_cos_t_star", ";cos(#theta*);Normalised counts", nbins, -1, 1)
    hfd_cos_t_star = TH1F("hfd_cos_t_star", ";cos(#theta*);Normalised counts", nbins, -1, 1)
    hbkg_dca = TH1F("hbkg_dca", ";DCA#kern[0.1]{ } (cm);Normalised counts", nbins, 0, 0.4)
    hpr_dca = TH1F("hpr_dca", ";DCA#kern[0.1]{ } (cm);Normalised counts", nbins, 0, 0.4)
    hfd_dca = TH1F("hfd_dca", ";DCA#kern[0.1]{ } (cm);Normalised counts", nbins, 0, 0.4)

    hbkg_ncombpi1 = TH1F("hbkg_ncombpi1", ";#it{N}_{sig}^{comb}(#pi pr1) (#sigma);Normalised counts", nbins, 0, 30)
    hpr_ncombpi1 = TH1F("hpr_ncombpi1", ";#it{N}_{sig}^{comb}(#pi pr1) (#sigma);Normalised counts", nbins, 0, 30)
    hfd_ncombpi1 = TH1F("hfd_ncombpi1", ";#it{N}_{sig}^{comb}(#pi pr1) (#sigma);Normalised counts", nbins, 0, 30)
    hbkg_ncombk1 = TH1F("hbkg_ncombk1", ";#it{N}_{sig}^{comb}(K pr1) (#sigma);Normalised counts", nbins, 0, 30)
    hpr_ncombk1 = TH1F("hpr_ncombk1", ";#it{N}_{sig}^{comb}(K pr1) (#sigma);Normalised counts", nbins, 0, 30)
    hfd_ncombk1 = TH1F("hfd_ncombk1", ";#it{N}_{sig}^{comb}(K pr1) (#sigma);Normalised counts", nbins, 0, 30)
    hbkg_ncombpi2 = TH1F("hbkg_ncombpi2", ";#it{N}_{sig}^{comb}(#pi pr2) (#sigma);Normalised counts", nbins, 0, 30)
    hpr_ncombpi2 = TH1F("hpr_ncombpi2", ";#it{N}_{sig}^{comb}(#pi pr2) (#sigma);Normalised counts", nbins, 0, 30)
    hfd_ncombpi2 = TH1F("hfd_ncombpi2", ";#it{N}_{sig}^{comb}(#pi pr2) (#sigma);Normalised counts", nbins, 0, 30)
    hbkg_ncombk2 = TH1F("hbkg_ncombk2", ";#it{N}_{sig}^{comb}(K pr2) (#sigma);Normalised counts", nbins, 0, 30)
    hpr_ncombk2 = TH1F("hpr_ncombk2", ";#it{N}_{sig}^{comb}(K pr2) (#sigma);Normalised counts", nbins, 0, 30)
    hfd_ncombk2 = TH1F("hfd_ncombk2", ";#it{N}_{sig}^{comb}(K pr2) (#sigma);Normalised counts", nbins, 0, 30)

    if BkgDf is not None:
        if 'd_len' in BkgDf.columns:
            BkgDf["d_len"] = BkgDf["d_len"].multiply(10000)
        if 'd_len_xy' in BkgDf.columns:
            BkgDf["d_len_xy"] = BkgDf["d_len_xy"].multiply(10000)
        if 'imp_par_xy' in BkgDf.columns:
            BkgDf["imp_par_xy"] = BkgDf["imp_par_xy"].multiply(10000)
            BkgDf["imp_par_xy"] = BkgDf["imp_par_xy"].abs()
        if 'delta_mass_D0' in BkgDf.columns:
            BkgDf["delta_mass_D0"] = BkgDf["delta_mass_D0"].multiply(1000)
        if 'max_norm_d0d0exp' in BkgDf.columns:
            BkgDf["max_norm_d0d0exp"] = BkgDf["max_norm_d0d0exp"].abs()
        if 'imp_par_prod' in BkgDf.columns:
            BkgDf["imp_par_prod"] = BkgDf["imp_par_prod"].multiply(10000*10000)

        if 'd_len' in BkgDf.columns:
            fill_hist(hbkg_d_len, BkgDf.d_len)
        if 'd_len_xy' in BkgDf.columns:
            fill_hist(hbkg_d_len_xy, BkgDf.d_len_xy)
        if 'norm_dl_xy' in BkgDf.columns:
            fill_hist(hbkg_norm_dl_xy, BkgDf.norm_dl_xy)
        if 'cos_p' in BkgDf.columns:
            fill_hist(hbkg_cos_p, BkgDf.cos_p)
        if 'cos_p_xy' in BkgDf.columns:
            fill_hist(hbkg_cos_p_xy, BkgDf.cos_p_xy)
        if 'imp_par_xy' in BkgDf.columns:
            fill_hist(hbkg_imp_par_xy, BkgDf.imp_par_xy)
        if 'cos_t_star' in BkgDf.columns:
            fill_hist(hbkg_cos_t_star, BkgDf.cos_t_star)
        if 'dca' in BkgDf.columns:
            fill_hist(hbkg_dca, BkgDf.dca)
        if 'delta_mass_D0' in BkgDf.columns:
            fill_hist(hbkg_delta_mass_D0, BkgDf.delta_mass_D0)
        if 'max_norm_d0d0exp' in BkgDf.columns:
            fill_hist(hbkg_max_norm_d0d0exp, BkgDf.max_norm_d0d0exp)
        if 'imp_par_prod' in BkgDf.columns:
            fill_hist(hbkg_imp_par_prod, BkgDf.imp_par_prod)

        if 'nsigComb_Pi_1' in BkgDf.columns:
            fill_hist(hbkg_ncombpi1, BkgDf.nsigComb_Pi_1)
        if 'nsigComb_K_1' in BkgDf.columns:
            fill_hist(hbkg_ncombk1, BkgDf.nsigComb_K_1)
        if 'nsigComb_Pi_2' in BkgDf.columns:
            fill_hist(hbkg_ncombpi2, BkgDf.nsigComb_Pi_2)
        if 'nsigComb_K_2' in BkgDf.columns:
            fill_hist(hbkg_ncombk2, BkgDf.nsigComb_K_2)

        print("Entries (bkg):", hbkg_d_len.GetEntries())
        
    if PromptDf is not None:
        if 'd_len' in PromptDf.columns:
            PromptDf["d_len"] = PromptDf["d_len"].multiply(10000)
        if 'd_len_xy' in PromptDf.columns:
            PromptDf["d_len_xy"] = PromptDf["d_len_xy"].multiply(10000)
        if 'imp_par_xy' in PromptDf.columns:
            PromptDf["imp_par_xy"] = PromptDf["imp_par_xy"].multiply(10000)
            PromptDf["imp_par_xy"] = PromptDf["imp_par_xy"].abs()
        if 'delta_mass_D0' in PromptDf.columns:
            PromptDf["delta_mass_D0"] = PromptDf["delta_mass_D0"].multiply(1000)
        if 'max_norm_d0d0exp' in PromptDf.columns:
            PromptDf["max_norm_d0d0exp"] = PromptDf["max_norm_d0d0exp"].abs()
        if 'imp_par_prod' in PromptDf.columns:
            PromptDf["imp_par_prod"] = PromptDf["imp_par_prod"].multiply(10000*10000)

        if 'd_len' in PromptDf.columns:
            fill_hist(hpr_d_len, PromptDf.d_len)
        if 'd_len_xy' in PromptDf.columns:
            fill_hist(hpr_d_len_xy, PromptDf.d_len_xy)
        if 'norm_dl_xy' in PromptDf.columns:
            fill_hist(hpr_norm_dl_xy, PromptDf.norm_dl_xy)
        if 'cos_p' in PromptDf.columns:
            fill_hist(hpr_cos_p, PromptDf.cos_p)
        if 'cos_p_xy' in PromptDf.columns:
            fill_hist(hpr_cos_p_xy, PromptDf.cos_p_xy)
        if 'imp_par_xy' in PromptDf.columns:
            fill_hist(hpr_imp_par_xy, PromptDf.imp_par_xy)
        if 'cos_t_star' in PromptDf.columns:
            fill_hist(hpr_cos_t_star, PromptDf.cos_t_star)
        if 'dca' in PromptDf.columns:
            fill_hist(hpr_dca, PromptDf.dca)
        if 'delta_mass_D0' in PromptDf.columns:
            fill_hist(hpr_delta_mass_D0, PromptDf.delta_mass_D0)
        if 'max_norm_d0d0exp' in PromptDf.columns:
            fill_hist(hpr_max_norm_d0d0exp, PromptDf.max_norm_d0d0exp)
        if 'imp_par_prod' in PromptDf.columns:
            fill_hist(hpr_imp_par_prod, PromptDf.imp_par_prod)

        if 'nsigComb_Pi_1' in PromptDf.columns:
            fill_hist(hpr_ncombpi1, PromptDf.nsigComb_Pi_1)
        if 'nsigComb_K_1' in PromptDf.columns:
            fill_hist(hpr_ncombk1, PromptDf.nsigComb_K_1)
        if 'nsigComb_Pi_2' in PromptDf.columns:
            fill_hist(hpr_ncombpi2, PromptDf.nsigComb_Pi_2)
        if 'nsigComb_K_2' in PromptDf.columns:
            fill_hist(hpr_ncombk2, PromptDf.nsigComb_K_2)

        print("Entries (prompt):", hpr_d_len.GetEntries())

    if FDDf is not None:
        if 'd_len' in FDDf.columns:
            FDDf["d_len"] = FDDf["d_len"].multiply(10000)
        if 'd_len_xy' in FDDf.columns:
            FDDf["d_len_xy"] = FDDf["d_len_xy"].multiply(10000)
        if 'imp_par_xy' in FDDf.columns:
            FDDf["imp_par_xy"] = FDDf["imp_par_xy"].multiply(10000)
            FDDf["imp_par_xy"] = FDDf["imp_par_xy"].abs()
        if 'delta_mass_D0' in FDDf.columns:
            FDDf["delta_mass_D0"] = FDDf["delta_mass_D0"].multiply(1000)
        if 'max_norm_d0d0exp' in FDDf.columns:
            FDDf["max_norm_d0d0exp"] = FDDf["max_norm_d0d0exp"].abs()
        if 'imp_par_prod' in FDDf.columns:
            FDDf["imp_par_prod"] = FDDf["imp_par_prod"].multiply(10000*10000)

        if 'd_len' in FDDf.columns:
            fill_hist(hfd_d_len, FDDf.d_len)
        if 'd_len_xy' in FDDf.columns:
            fill_hist(hfd_d_len_xy, FDDf.d_len_xy)
        if 'norm_dl_xy' in FDDf.columns:
            fill_hist(hfd_norm_dl_xy, FDDf.norm_dl_xy)
        if 'cos_p' in FDDf.columns:
            fill_hist(hfd_cos_p, FDDf.cos_p)
        if 'cos_p_xy' in FDDf.columns:
            fill_hist(hfd_cos_p_xy, FDDf.cos_p_xy)
        if 'imp_par_xy' in FDDf.columns:
            fill_hist(hfd_imp_par_xy, FDDf.imp_par_xy)
        if 'cos_t_star' in FDDf.columns:
            fill_hist(hfd_cos_t_star, FDDf.cos_t_star)
        if 'dca' in FDDf.columns:
            fill_hist(hfd_dca, FDDf.dca)
        if 'delta_mass_D0' in FDDf.columns:
            fill_hist(hfd_delta_mass_D0, FDDf.delta_mass_D0)
        if 'max_norm_d0d0exp' in FDDf.columns:
            fill_hist(hfd_max_norm_d0d0exp, FDDf.max_norm_d0d0exp)
        if 'imp_par_prod' in FDDf.columns:
            fill_hist(hfd_imp_par_prod, FDDf.imp_par_prod)

        if 'nsigComb_Pi_1' in FDDf.columns:
            fill_hist(hfd_ncombpi1, FDDf.nsigComb_Pi_1)
        if 'nsigComb_K_1' in FDDf.columns:
            fill_hist(hfd_ncombk1, FDDf.nsigComb_K_1)
        if 'nsigComb_Pi_2' in FDDf.columns:
            fill_hist(hfd_ncombpi2, FDDf.nsigComb_Pi_2)
        if 'nsigComb_K_2' in FDDf.columns:
            fill_hist(hfd_ncombk2, FDDf.nsigComb_K_2)

        print("Entries (feed-down):", hfd_d_len.GetEntries())

    out_file.cd()
    if 'd_len' in vars_to_plot:
        hbkg_d_len.Write()
    if 'd_len' in vars_to_plot:
        hpr_d_len.Write()
    if 'd_len' in vars_to_plot:
        hfd_d_len.Write()
    if 'd_len_xy' in vars_to_plot:
        hbkg_d_len_xy.Write()
    if 'd_len_xy' in vars_to_plot:
        hpr_d_len_xy.Write()
    if 'd_len_xy' in vars_to_plot:
        hfd_d_len_xy.Write()
    if 'norm_dl_xy' in vars_to_plot:
        hbkg_norm_dl_xy.Write()
    if 'norm_dl_xy' in vars_to_plot:
        hpr_norm_dl_xy.Write()
    if 'norm_dl_xy' in vars_to_plot:
        hfd_norm_dl_xy.Write()
    if 'cos_p' in vars_to_plot:
        hbkg_cos_p.Write()
    if 'cos_p' in vars_to_plot:
        hpr_cos_p.Write()
    if 'cos_p' in vars_to_plot:
        hfd_cos_p.Write()
    if 'cos_p_xy' in vars_to_plot:
        hbkg_cos_p_xy.Write()
    if 'cos_p_xy' in vars_to_plot:
        hpr_cos_p_xy.Write()
    if 'cos_p_xy' in vars_to_plot:
        hfd_cos_p_xy.Write()
    if 'imp_par_xy' in vars_to_plot:
        hbkg_imp_par_xy.Write()
    if 'imp_par_xy' in vars_to_plot:
        hpr_imp_par_xy.Write()
    if 'imp_par_xy' in vars_to_plot:
        hfd_imp_par_xy.Write()

    if 'delta_mass_D0' in vars_to_plot:
        hbkg_delta_mass_D0.Write()
    if 'delta_mass_D0' in vars_to_plot:
        hpr_delta_mass_D0.Write()
    if 'delta_mass_D0' in vars_to_plot:
        hfd_delta_mass_D0.Write()
    if 'max_norm_d0d0exp' in vars_to_plot:
        hbkg_max_norm_d0d0exp.Write()
    if 'max_norm_d0d0exp' in vars_to_plot:
        hpr_max_norm_d0d0exp.Write()
    if 'max_norm_d0d0exp' in vars_to_plot:
        hfd_max_norm_d0d0exp.Write()
    if 'imp_par_prod' in vars_to_plot:
        hbkg_imp_par_prod.Write()
    if 'imp_par_prod' in vars_to_plot:
        hpr_imp_par_prod.Write()
    if 'imp_par_prod' in vars_to_plot:
        hfd_imp_par_prod.Write()
    if 'cos_t_star' in vars_to_plot:
        hbkg_cos_t_star.Write()
    if 'cos_t_star' in vars_to_plot:
        hpr_cos_t_star.Write()
    if 'cos_t_star' in vars_to_plot:
        hfd_cos_t_star.Write()
    if 'dca' in vars_to_plot:
        hbkg_dca.Write()
    if 'dca' in vars_to_plot:
        hpr_dca.Write()
    if 'dca' in vars_to_plot:
        hfd_dca.Write()

    if 'nsigComb_Pi_1' in vars_to_plot:
        hbkg_ncombpi1.Write()
    if 'nsigComb_Pi_1' in vars_to_plot:
        hpr_ncombpi1.Write()
    if 'nsigComb_Pi_1' in vars_to_plot:
        hfd_ncombpi1.Write()
    if 'nsigComb_K_1' in vars_to_plot:
        hbkg_ncombk1.Write()
    if 'nsigComb_K_1' in vars_to_plot:
        hpr_ncombk1.Write()
    if 'nsigComb_K_1' in vars_to_plot:
        hfd_ncombk1.Write()
    if 'nsigComb_Pi_2' in vars_to_plot:
        hbkg_ncombpi2.Write()
    if 'nsigComb_Pi_2' in vars_to_plot:
        hpr_ncombpi2.Write()
    if 'nsigComb_Pi_2' in vars_to_plot:
        hfd_ncombpi2.Write()
    if 'nsigComb_K_2' in vars_to_plot:
        hbkg_ncombk2.Write()
    if 'nsigComb_K_2' in vars_to_plot:
        hpr_ncombk2.Write()
    if 'nsigComb_K_2' in vars_to_plot:
        hfd_ncombk2.Write()

    out_file.Close()

