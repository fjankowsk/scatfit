#
#   Plotting functions.
#   2022 - 2025 Fabian Jankowski
#

import corner
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from lmfit import Model

from scatfit.dm import get_dm_smearing
import scatfit.pulsemodels as pulsemodels


def use_custom_matplotlib_formatting():
    """
    Adjust the matplotlib configuration parameters for custom format.
    """

    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = 14.0
    matplotlib.rcParams["lines.markersize"] = 8
    matplotlib.rcParams["legend.frameon"] = False
    # make tickmarks more visible
    matplotlib.rcParams["xtick.major.size"] = 6
    matplotlib.rcParams["xtick.major.width"] = 1.5
    matplotlib.rcParams["xtick.minor.size"] = 4
    matplotlib.rcParams["xtick.minor.width"] = 1.5
    matplotlib.rcParams["ytick.major.size"] = 6
    matplotlib.rcParams["ytick.major.width"] = 1.5
    matplotlib.rcParams["ytick.minor.size"] = 4
    matplotlib.rcParams["ytick.minor.width"] = 1.5


def plot_frb(cand, plot_range, profile, params):
    """
    Plot the FRB data.
    """

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex="col",
        gridspec_kw={"height_ratios": [0.5, 1], "hspace": 0},
    )

    ax1.step(
        plot_range,
        profile,
        where="mid",
        color="black",
        ls="solid",
        lw=1.0,
        zorder=3,
    )

    # highlight the data included in the fit
    for item in params["fitrange"]:
        ax1.axvline(x=item, color="tab:red", zorder=4)

    ax1.set_ylabel("Flux\n(a.u.)")
    ax1.tick_params(bottom=False)

    # plot dedispersed waterfall
    freqs = cand.freqs
    chan_bw = np.diff(freqs)[0]

    quantiles = np.quantile(
        cand.dynspec,
        q=[0.05, 0.1, 0.25, 0.3, 0.4, 0.5, 0.75, 0.8, 0.95, 0.97, 0.98, 0.99],
    )

    ax2.imshow(
        cand.dynspec,
        aspect="auto",
        interpolation=None,
        extent=[
            plot_range[0],
            plot_range[-1],
            freqs[-1] - 0.5 * chan_bw,
            freqs[0] + 0.5 * chan_bw,
        ],
        cmap="Greys",
        vmin=quantiles[0],
        vmax=quantiles[-4],
        zorder=3,
    )

    # highlight the data included in the fit
    for item in params["fitrange"]:
        ax2.axvline(x=item, color="tab:red", zorder=4)

    ax2.set_ylabel("Frequency\n(MHz)")
    ax2.tick_params(bottom=False)

    # align ylabels horizontally
    fig.align_labels()

    fig.tight_layout()


def plot_frb_scat(
    cand,
    df,
    fitresults,
    smodel,
    plot_range,
    params,
    cmap1="Greys",
    cmap2="YlGnBu",
    dynspec=True,
):
    """
    Plot the dynamic spectrum and the sub-band profiles in a combined figure.
    """

    # TODO: Include additional arguments in result to be ablo to plot all models
    if smodel == "unscattered":
        scat_model = pulsemodels.gaussian_normed
    elif smodel == "scattered_isotropic_analytic":
        scat_model = pulsemodels.scattered_gaussian_pulse
    elif smodel == "scattered_isotropic_convolving":
        scat_model = pulsemodels.scattered_profile
    elif smodel == "scattered_isotropic_bandintegrated":
        scat_model = pulsemodels.bandintegrated_model
    elif smodel == "scattered_isotropic_afb_instrumental":
        scat_model = pulsemodels.gaussian_scattered_afb_instrumental
    elif smodel == "scattered_isotropic_dfb_instrumental":
        scat_model = pulsemodels.gaussian_scattered_dfb_instrumental
    else:
        raise NotImplementedError(f"Scattering model not implemented: {smodel}")

    model = Model(scat_model)

    cmap1 = plt.get_cmap(cmap1)
    color1 = [cmap1((ii + 2) / (df.shape[0] + 2)) for ii in range(df.shape[0])]
    cmap2 = plt.get_cmap(cmap2)
    color2 = [cmap2((ii + 2) / (df.shape[0] + 2)) for ii in range(df.shape[0])]

    # setting up plot
    if dynspec:
        fig, axs = plt.subplots(
            figsize=(7, 7),
            nrows=2,
            ncols=1,
            sharex=True,
            gridspec_kw={"height_ratios": [2, 3], "hspace": 0},
        )
    else:
        fig, axs = plt.subplots(figsize=(7, 3), nrows=1, ncols=1)
        axs = [axs]

    for iband, row in df.iterrows():
        band = int(row["band"])

        sub_profile = cand.dynspec[band, :]
        sub_profile = sub_profile - np.mean(sub_profile)
        sub_profile = sub_profile / np.max(sub_profile)

        fitresult = fitresults[iband]
        axs[0].plot(
            plot_range,
            model.eval(params=fitresult.params, x=plot_range) - band,
            color=color2[iband],
            lw=1.5,
            zorder=8,
        )
        axs[0].plot(
            plot_range, sub_profile - band, color=color1[iband], lw=0.5, alpha=0.7
        )

    yloc = -df["band"].to_numpy()
    labels = [str(int(f)) for f in df["cfreq"].to_numpy()]
    axs[0].set_yticks(yloc, labels=labels)

    axs[0].set_xlim(left=params["zoom"][0], right=params["zoom"][1])
    if not dynspec:
        axs[0].set_ylabel("Frequency (MHz)")
        axs[0].set_xlabel("Time (ms)")

    axs[0].tick_params(
        axis="both",
        which="both",
        direction="in",
        bottom=True,
        top=True,
        left=True,
        right=True,
    )

    if dynspec:
        # remove bottom labels (numbers)
        axs[0].tick_params(axis="x", which="both", labelbottom=False)

        freqs = cand.freqs
        chan_bw = np.diff(freqs)[0]

        axs[1].imshow(
            cand.dynspec,
            aspect="auto",
            interpolation=None,
            extent=[
                plot_range[0],
                plot_range[-1],
                freqs[-1] - 0.5 * chan_bw,
                freqs[0] + 0.5 * chan_bw,
            ],
            cmap=cmap2,
            vmin=np.percentile(cand.dynspec, 0.1),
            vmax=np.percentile(cand.dynspec, 99.9),
        )

        axs[1].set_xlim(left=params["zoom"][0], right=params["zoom"][1])
        axs[1].set_xlabel("Time (ms)")

        # use a common ylabel
        fig.supylabel("Frequency (MHz)")

        axs[1].tick_params(
            axis="both",
            which="both",
            direction="in",
            bottom=True,
            top=True,
            left=True,
            right=True,
        )

    fig.tight_layout()

    if dynspec:
        filename = "scattering_fit_allbands_dynspec.pdf"
    else:
        filename = "scattering_fit_allbands.pdf"

    fig.savefig(filename, bbox_inches="tight", pad_inches=0.1)


def plot_profile_models(params):
    """
    Plot and compare the profile scattering models.

    Parameters
    ----------
    params: dict
        Additional parameters that affect the processing.
    """

    plot_range = np.linspace(-200.0, 200.0, num=8000)

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(
        plot_range,
        pulsemodels.gaussian_normed(plot_range, 5.0, 10.0, 1.5),
        lw=2,
        label="unscattered",
        zorder=4,
    )

    ax.plot(
        plot_range,
        pulsemodels.scattered_gaussian_pulse(plot_range, 5.0, 0.0, 1.5, 2.5, 0.0),
        lw=2,
        label="analytic",
        zorder=5,
    )

    ax.plot(
        plot_range,
        pulsemodels.scattered_profile(plot_range, 5.0, 0.0, 1.5, 2.5, 0.0),
        lw=2,
        label="convolving",
        zorder=4,
    )

    ax.plot(
        plot_range,
        pulsemodels.scattered_profile(plot_range, 5.0, 20.0, 1.5, 2.5, 0.0),
        lw=2,
        label="convolving",
        zorder=4,
    )

    ax.plot(
        plot_range,
        pulsemodels.bandintegrated_model(
            plot_range, 5.0, 10.0, 1.5, 2.5, 0.0, 856.0, 1712.0, 21
        ),
        lw=2,
        label="bandintegrated",
        zorder=5,
    )

    ax.plot(
        plot_range,
        pulsemodels.gaussian_scattered_afb_instrumental(
            plot_range, 5.0, 0.0, 1.5, 2.5, 0.306, 2.0, 0.0
        ),
        lw=2,
        label="afb",
        zorder=4,
    )

    ax.plot(
        plot_range,
        pulsemodels.gaussian_scattered_dfb_instrumental(
            plot_range, 5.0, 0.0, 1.5, 2.5, 4.0, 0.0
        ),
        lw=2,
        label="dfb",
        zorder=4,
    )

    ax.grid()
    ax.legend(loc="best")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Flux (a.u.)")

    fig.tight_layout()

    if params["output"]:
        fig.savefig("profile_models.pdf", bbox_inches="tight")


def plot_profile_fit(
    fit_range, sub_profile, fitresult, iband, cfreq, params, fitresult2=None
):
    """
    Plot the profile fit.
    """

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex="col",
        gridspec_kw={"height_ratios": [1, 0.3], "hspace": 0},
    )

    ax1.step(
        fit_range,
        sub_profile,
        where="mid",
        color="black",
        ls="solid",
        lw=1.0,
        zorder=3,
    )

    if not params["publish"]:
        ax1.plot(
            fit_range,
            fitresult.init_fit,
            color="tab:orange",
            ls="dotted",
            lw=1.5,
            zorder=6,
        )

    ax1.plot(
        fit_range,
        fitresult.best_fit,
        color="tab:red",
        ls="dashed",
        lw=2.0,
        zorder=8,
    )

    # plot second fit
    if fitresult2 is not None:
        ax1.plot(
            fit_range,
            fitresult2.best_fit,
            color="tab:blue",
            ls="dotted",
            lw=2.0,
            zorder=6,
        )

    # show centre frequency and scattering time
    info_str = f"{cfreq:.0f} MHz"

    if "taus" in fitresult.best_values:
        info_str += "\n" + "${0:.1f} \pm {1:.1f}$ ms".format(
            fitresult.best_values["taus"], fitresult.params["taus"].stderr
        )

    ax1.text(
        x=0.025,
        y=1 - 0.04,
        s=info_str,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax1.transAxes,
        zorder=8,
    )

    ax1.set_ylabel("Flux (a.u.)")
    if not params["publish"]:
        ax1.set_title(f"Sub-band {iband}")

    # hide bottom ticks
    ax1.tick_params(bottom=False)

    # residuals
    residual = sub_profile - fitresult.best_fit

    ax2.step(
        fit_range,
        residual,
        where="mid",
        color="black",
        ls="solid",
        lw=1.0,
        zorder=3,
    )

    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Residual")

    # zoom in
    ax2.set_xlim(left=params["zoom"][0], right=params["zoom"][1])

    # align the labels of the subplots horizontally and vertically
    fig.align_labels()

    fig.tight_layout()

    fig.savefig(f"scattering_fit_band-{iband}.pdf", bbox_inches="tight")


def plot_width_scaling(t_df, cand, fitresult, params):
    """
    Plot the scaling of fitted widths with frequency.
    """

    df = t_df.copy()

    freqs = cand.freqs
    chan_bw = np.diff(freqs)[0]

    # switch between ghz and mhz
    if np.max(freqs) > 1000.0:
        fact = 1e-3
        xlabel = "Frequency (GHz)"
    else:
        fact = 1
        xlabel = "Frequency (MHz)"

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.errorbar(
        x=fact * df["cfreq"],
        y=df["w50i"],
        yerr=df["err_w50i"],
        fmt="o",
        color="black",
        zorder=7,
        label=r"$\mathrm{W}_\mathrm{50i}$",
    )

    ax.errorbar(
        x=fact * df["cfreq"],
        y=df["w10i"],
        yerr=df["err_w10i"],
        fmt="+",
        color="darkgrey",
        zorder=6,
        label=r"$\mathrm{W}_\mathrm{10i}$",
    )

    ax.scatter(
        x=fact * df["cfreq"],
        y=df["weq"],
        color="grey",
        marker="*",
        zorder=8,
        label=r"$\mathrm{W}_\mathrm{eq}$",
    )

    ax.scatter(
        x=fact * df["cfreq"],
        y=df["w50p"],
        color="grey",
        marker="d",
        zorder=9,
        label=r"$\mathrm{W}_\mathrm{50p}$",
    )

    ax.scatter(
        x=fact * df["cfreq"],
        y=df["w10p"],
        color="grey",
        marker="+",
        zorder=9,
        label=r"$\mathrm{W}_\mathrm{10p}$",
    )

    if "taus" in df.columns:
        ax.errorbar(
            x=fact * df["cfreq"],
            y=df["taus"],
            yerr=df["err_taus"],
            fmt="x",
            color="dimgrey",
            zorder=4,
            label=r"$\tau_\mathrm{s}$",
        )

    # scattering time fit
    if fitresult is not None:
        ax.plot(
            fact * df["cfreq"],
            10**fitresult.best_fit,
            color="dimgrey",
            ls="solid",
            lw=2.0,
            zorder=3.5,
            label=r"$\tau_\mathrm{s}$ fit",
        )

        # show scattering time at 1 ghz or 100 mhz and scattering index
        if df["cfreq"].mean() < 400:
            info_str = "${0:.1f} \pm {1:.1f}$ ms".format(
                fitresult.tau_100mhz["value"], fitresult.tau_100mhz["error"]
            )
        else:
            info_str = "${0:.1f} \pm {1:.1f}$ ms".format(
                fitresult.tau_1ghz["value"], fitresult.tau_1ghz["error"]
            )
        info_str += "\n" + "${0:.1f} \pm {1:.1f}$".format(
            fitresult.best_values["slope"], fitresult.params["slope"].stderr
        )

        ax.text(
            x=1 - 0.025,
            y=1 - 0.04,
            s=info_str,
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            zorder=8,
        )

    # intra-channel dispersive smearing
    if not params["nodmsmearing"]:
        f_lo = np.sort(freqs)
        f_hi = f_lo + np.abs(chan_bw)

        dm_smear = get_dm_smearing(f_lo, f_hi, cand.dm)

        plot_range = 0.5 * (f_lo + f_hi)

        ax.plot(
            fact * plot_range,
            dm_smear,
            color="grey",
            ls="dashed",
            lw=2.0,
            zorder=3,
            label=r"$t_\mathrm{dm}$",
        )

    ax.grid()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Width (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")

    sfor = FormatStrFormatter("%g")
    ax.xaxis.set_major_formatter(sfor)
    ax.xaxis.set_minor_formatter(sfor)
    ax.yaxis.set_major_formatter(sfor)

    fig.tight_layout()

    fig.savefig("width_scaling.pdf", bbox_inches="tight")


def plot_center_scaling(t_df, params):
    """
    Plot the scaling of fitted center with frequency.

    Parameters
    ----------
    t_df: ~pd.DataFrame
        The input data.
    params: dict
        Other parameters that affect the output.
    """

    df = t_df.copy()

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.errorbar(
        x=1e-3 * df["cfreq"],
        y=df["center"],
        yerr=df["err_center"],
        fmt="o",
        color="black",
        zorder=5,
    )

    ax.grid()
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Center (ms)")

    fig.tight_layout()

    if params["output"]:
        fig.savefig("center_scaling.pdf", bbox_inches="tight")


def plot_chains(fitresult_emcee, params):
    """
    Plot the MCMC chains from an emcee sampling run.

    Parameters
    ----------
    fitresult_emcee: ~lmfit.MinimizerResult
        The minimizer result object from lmfit.
    params: dict
        Other parameters that affect the output.
    """

    samples = fitresult_emcee.flatchain
    var_names = fitresult_emcee.var_names
    nvary = len(var_names)

    fig, axs = plt.subplots(nrows=nvary, ncols=1, sharex=True, figsize=(8, 9))

    for i, name in enumerate(var_names):
        axs[i].plot(samples.iloc[:, i], color="black", alpha=0.5)
        axs[i].set_ylabel(name)

    axs[nvary - 1].set_xlabel("Step Number")

    if "iband" in params:
        fig.suptitle("Sub-band {0}".format(params["iband"]))

    # align the labels of the subplots vertically
    fig.align_ylabels()

    fig.tight_layout()

    if params["output"]:
        filename = "chains_band-{0}.pdf".format(params["iband"])
        fig.savefig(filename, bbox_inches="tight")


def plot_corner(fitresult_emcee, fnlabel, params):
    """
    Make a corner plot.

    Parameters
    ----------
    fitresult_emcee: ~lmfit.MinimizerResult
        The minimizer result object from lmfit.
    fnlabel: str
        Postfix label for the filename.
    params: dict
        Additional plotting parameters.
    """

    samples = fitresult_emcee.flatchain

    # get maximum likelihood values
    max_likelihood = np.argmax(fitresult_emcee.lnprob)
    max_likelihood_idx = np.unravel_index(max_likelihood, fitresult_emcee.lnprob.shape)
    max_likelihood_values = fitresult_emcee.chain[max_likelihood_idx]

    # defaults
    bins = 20
    fontsize_before = matplotlib.rcParams["font.size"]
    hist_kwargs = None
    labelpad = 0.125
    max_n_ticks = 5
    plot_datapoints = False
    show_titles = True
    smooth = False
    var_names = fitresult_emcee.var_names

    if not params["fast"]:
        bins = 40

    if params["publish"]:
        hist_kwargs = {"lw": 2.0}
        labelpad = 0.475
        max_n_ticks = 2
        matplotlib.rcParams["font.size"] = 34.0
        show_titles = False
        smooth = True

        mapping = {
            "fluence": "$F$",
            "center": "$t_0$",
            "sigma": r"$\sigma$",
            "taus": r"$\tau_s$",
            "dc": "$b$",
            "__lnsigma": r"$\ln(\epsilon)$",
        }

        for idx, key in enumerate(var_names):
            if key in mapping:
                var_names[idx] = mapping[key]

    fig = corner.corner(
        samples,
        bins=bins,
        hist_kwargs=hist_kwargs,
        labels=var_names,
        labelpad=labelpad,
        max_n_ticks=max_n_ticks,
        truths=max_likelihood_values,
        plot_datapoints=plot_datapoints,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=show_titles,
        smooth=smooth,
        title_kwargs={"fontsize": 10},
    )

    if "iband" in params and not params["publish"]:
        fig.suptitle("Sub-band {0}".format(params["iband"]))

    if params["output"]:
        if "iband" in params:
            filename = "corner_{0}_band-{1}.pdf".format(fnlabel, params["iband"])
        else:
            filename = f"corner_{fnlabel}.pdf"

        fig.savefig(filename, bbox_inches="tight")

    # reset
    matplotlib.rcParams["font.size"] = fontsize_before
