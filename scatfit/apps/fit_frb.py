#
#   Fit scattering models to FRB data.
#   2022 - 2024 Fabian Jankowski
#   2023 Ines Pastor-Marazuela
#

import argparse
import copy
import inspect
import os.path
import sys

from astropy.time import Time, TimeDelta
from lmfit import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scatfit.dm import KDM, get_dm_smearing
import scatfit.plotting as plotting
import scatfit.pulsemodels as pulsemodels
import scatfit.sigproc as sigproc
import scatfit.archive as archive


def parse_args():
    """
    Parse the commandline arguments.

    Returns
    -------
    args: populated namespace
        The commandline arguments.
    """

    parser = argparse.ArgumentParser(
        description="Fit a scattering model to FRB data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "filename", type=str, help="The name of the input filterbank file."
    )

    parser.add_argument(
        "dm",
        type=float,
        help="The dispersion measure of the FRB.",
    )

    parser.add_argument(
        "--compare",
        dest="compare",
        action="store_true",
        default=False,
        help="Fit an unscattered Gaussian model for comparison.",
    )

    parser.add_argument(
        "--binburst",
        dest="bin_burst",
        default=None,
        metavar="bin",
        type=int,
        help="Specify the burst location bin manually.",
    )

    parser.add_argument(
        "--fscrunch",
        dest="fscrunch_factor",
        default=256,
        metavar="factor",
        type=int,
        help="Integrate this many frequency channels.",
    )

    parser.add_argument(
        "--tscrunch",
        dest="tscrunch_factor",
        default=1,
        metavar="factor",
        type=int,
        help="Integrate this many time samples.",
    )

    parser.add_argument(
        "--fast",
        dest="fast",
        action="store_true",
        default=False,
        help="Enable fast processing. This reduces the number of MCMC steps drastically.",
    )

    parser.add_argument(
        "--fitrange",
        dest="fitrange",
        type=float,
        nargs=2,
        metavar=("start", "end"),
        default=[-150.0, 150.0],
        help="Consider only this time range of data in the fit. Increase the region for wide or highly-scattered bursts. Ensure that most of the scattering tail is included in the fit.",
    )

    parser.add_argument(
        "--fitscatindex",
        action="store_true",
        dest="fit_scatindex",
        default=False,
        help="Fit the scattering times and determine the scattering index.",
    )

    parser.add_argument(
        "--norfi",
        action="store_true",
        dest="norfi",
        default=False,
        help="Disable all internal RFI excision methods and use the input data as provided (aside from scaling). This is useful for synthetic input data or if you have cleaned the data already using external tools.",
    )

    parser.add_argument(
        "--smodel",
        dest="smodel",
        choices=[
            "unscattered",
            "scattered_isotropic_analytic",
            "scattered_isotropic_convolving",
            "scattered_isotropic_bandintegrated",
            "scattered_isotropic_afb_instrumental",
            "scattered_isotropic_dfb_instrumental",
        ],
        default="scattered_isotropic_analytic",
        help="Use the specified scattering model.",
    )

    parser.add_argument(
        "--showmodels",
        action="store_true",
        dest="show_models",
        default=False,
        help="Show comparison plot of scattering models.",
    )

    parser.add_argument(
        "--snr",
        dest="snr",
        default=3.8,
        metavar="snr",
        type=float,
        help="Only consider sub-bands above this S/N threshold.",
    )

    parser.add_argument(
        "--publish",
        dest="publish",
        action="store_true",
        default=False,
        help="Output plots suitable for publication.",
    )

    parser.add_argument(
        "-z",
        "--zoom",
        dest="zoom",
        type=float,
        nargs=2,
        metavar=("start", "end"),
        default=[-50.0, 50.0],
        help="Zoom into this time region.",
    )

    args = parser.parse_args()

    return args


def check_args(args):
    """
    Sanity check the commandline arguments.

    Parameters
    ----------
    args: populated namespace
        The commandline arguments.
    """

    # fitrange
    if args.fitrange[0] < args.fitrange[1]:
        pass
    else:
        print(
            "Fit range is invalid: {0}, {1}".format(args.fitrange[0], args.fitrange[1])
        )
        sys.exit(1)

    # zoom
    if args.zoom[0] < args.zoom[1]:
        pass
    else:
        print("Zoom range is invalid: {0}, {1}".format(args.zoom[0], args.zoom[1]))
        sys.exit(1)


def linear(x, x0, slope, intercept):
    """
    A linear function.

    Parameters
    ----------
    x: ~np.array
        The running variable.
    x0: float
        The reference location.
    slope: float
        The slope of the line.
    intercept: float
        The y value at x0.

    Returns
    -------
    res: ~np.array
        The model data.
    """

    res = slope * (x - x0) + intercept

    return res


def fit_powerlaw(x, y, err_y, params):
    """
    Fit a power law to data.
    """

    # convert to logscale
    log_x = np.log10(x)
    log_y = np.log10(y)
    log_err_y = err_y / (np.log(10) * y)

    model = Model(linear)

    model.set_param_hint("x0", value=np.log10(np.mean(x)), vary=False)
    model.set_param_hint("slope", value=-4.4)
    model.set_param_hint("intercept", value=np.log10(np.mean(y)))

    fitparams = model.make_params()

    fitresult_ml = model.fit(
        data=log_y, x=log_x, weights=1.0 / log_err_y, params=fitparams, method="leastsq"
    )

    if not fitresult_ml.success:
        raise RuntimeError("Fit did not converge.")

    print(fitresult_ml.fit_report())

    # 100 * (2000 - 700)/10 = 13k samples
    emcee_kws = dict(steps=2000, burn=700, thin=10, is_weighted=True, progress=True)

    emcee_params = fitresult_ml.params.copy()

    fitresult_emcee = model.fit(
        data=log_y,
        x=log_x,
        weights=1.0 / log_err_y,
        params=emcee_params,
        method="emcee",
        fit_kws=emcee_kws,
    )

    print(fitresult_emcee.fit_report())

    plotting.plot_corner(fitresult_emcee, "", False, params)

    print(
        "Scattering index: {0:.2f} +- {1:.2f}".format(
            fitresult_emcee.best_values["slope"], fitresult_emcee.params["slope"].stderr
        )
    )

    # estimate scattering time at 1 ghz and 100 MHz from the mcmc samples
    samples = fitresult_emcee.flatchain

    tau_rfreqs = []

    for rfreq in [1.0, 0.1]:
        tau_samples = 10 ** linear(
            np.log10(rfreq),
            fitresult_emcee.best_values["x0"],
            samples["slope"],
            samples["intercept"],
        )

        quantiles = np.quantile(tau_samples, q=[0.16, 0.5, 0.84])

        error = np.maximum(
            np.abs(quantiles[1] - quantiles[0]), np.abs(quantiles[2] - quantiles[1])
        )

        tau_rfreq = {"rfreq": rfreq, "value": quantiles[1], "error": error}

        print(
            "Scattering time at {0} GHz: {1:.2f} +- {2:.2f} ms".format(
                tau_rfreq["rfreq"], tau_rfreq["value"], tau_rfreq["error"]
            )
        )

        tau_rfreqs.append(tau_rfreq)

    # store the computed values in the result object
    fitresult_emcee.tau_1ghz = tau_rfreqs[0]
    fitresult_emcee.tau_100mhz = tau_rfreqs[1]

    return fitresult_emcee


def compute_updated_dm(t_df, dm, params):
    """
    Compute an updated dispersion measure by fitting the
    center versus frequency curve.
    """

    df = t_df.copy()

    # x is in MHz^-2
    x = (df["cfreq"] ** -2 - df["cfreq"].iat[0] ** -2).to_numpy()
    y = (df["center"] - df["center"].iat[0]).to_numpy()
    err_y = df["err_center"].to_numpy()

    # convert to seconds and divide by dispersion constant
    # so the slope is delta dm in the right units
    y = 1e-3 * y / KDM
    err_y = 1e-3 * err_y / KDM

    model = Model(linear)

    model.set_param_hint("x0", value=np.mean(x), vary=False)
    model.set_param_hint("slope", value=0.0)
    model.set_param_hint("intercept", value=np.mean(y))

    fitparams = model.make_params()

    fitresult_ml = model.fit(
        data=y, x=x, weights=1.0 / err_y, params=fitparams, method="leastsq"
    )

    if not fitresult_ml.success:
        raise RuntimeError("Fit did not converge.")

    print(fitresult_ml.fit_report())

    # 100 * (2000 - 700)/10 = 13k samples
    emcee_kws = dict(steps=2000, burn=700, thin=10, is_weighted=True, progress=True)

    emcee_params = fitresult_ml.params.copy()

    fitresult_emcee = model.fit(
        data=y,
        x=x,
        weights=1.0 / err_y,
        params=emcee_params,
        method="emcee",
        fit_kws=emcee_kws,
    )

    print(fitresult_emcee.fit_report())

    plotting.plot_corner(fitresult_emcee, "", False, params)

    delta_dm = fitresult_emcee.best_values["slope"]
    err_delta_dm = fitresult_emcee.params["slope"].stderr

    updated_dm = {"value": dm + delta_dm, "error": err_delta_dm}

    print(
        "Updated DM: {0:.4f} +- {1:.4f} pc cm^-3".format(
            updated_dm["value"], updated_dm["error"]
        )
    )


def fit_profile_model(fit_range, profile, smodel, params):
    """
    Fit a profile model to data.
    """

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
        raise NotImplementedError(
            "Scattering model not implemented: {0}".format(smodel)
        )

    model = Model(scat_model)

    model.set_param_hint("fluence", value=5.0, min=0.1)
    model.set_param_hint("center", value=0.0, min=-20.0, max=20.0)
    # XXX: we should set the minimum based on the sampling time of the data
    # model.set_param_hint("sigma", value=1.5, min=0.30624, max=20.0)
    model.set_param_hint("sigma", value=1.5, min=7.0e-5, max=20.0)

    arg_list = list(inspect.signature(scat_model).parameters.keys())

    if "taus" in arg_list:
        # XXX: same here
        # model.set_param_hint("taus", value=1.5, min=0.1)
        model.set_param_hint("taus", value=1.5, min=5.0e-5)

    if "taui" in arg_list:
        # XXX: same here
        model.set_param_hint("taui", value=0.30624, vary=False)

    if "taud" in arg_list:
        model.set_param_hint("taud", value=params["dm_smear"], vary=False)

    if "dc" in arg_list:
        model.set_param_hint("dc", value=0.0, min=-0.3, max=0.3)

    if "f_lo" in arg_list:
        model.set_param_hint("f_lo", value=params["f_lo"], vary=False)

    if "f_hi" in arg_list:
        model.set_param_hint("f_hi", value=params["f_hi"], vary=False)

    if "nfreq" in arg_list:
        if params["fast"]:
            model.set_param_hint("nfreq", value=3, vary=False)
        else:
            model.set_param_hint("nfreq", value=9, vary=False)

    fitparams = model.make_params()

    fitparams.add("w50i", expr="2.3548200*sigma")
    fitparams.add("w10i", expr="4.2919320*sigma")

    fitresult_ml = model.fit(
        data=profile, x=fit_range, params=fitparams, method="leastsq"
    )

    if not fitresult_ml.success:
        raise RuntimeError("Fit did not converge.")

    print(fitresult_ml.fit_report())

    # 100 * (5000 - 1000)/10 = 40k samples
    emcee_kws = dict(steps=5000, burn=1000, thin=10, is_weighted=False, progress=True)

    if params["fast"]:
        # 100 * (300 - 100)/2 = 10k samples
        emcee_kws["steps"] = 300
        emcee_kws["burn"] = 100
        emcee_kws["thin"] = 2

    emcee_params = fitresult_ml.params.copy()
    emcee_params.add("__lnsigma", value=np.log(0.1), min=np.log(0.001), max=np.log(2.0))

    fitresult_emcee = model.fit(
        data=profile,
        x=fit_range,
        params=emcee_params,
        method="emcee",
        fit_kws=emcee_kws,
    )

    print(fitresult_emcee.fit_report())

    plotting.plot_chains(fitresult_emcee)
    plotting.plot_corner(fitresult_emcee, smodel, True, params)

    return fitresult_emcee


def compute_post_widths(fit_range, t_fitresult):
    """
    Compute the full post-scattering widths numerically
    from the Markov chain samples.
    """

    fitresult = copy.deepcopy(t_fitresult)

    # oversample the model
    dense_range = np.linspace(
        np.min(fit_range), np.max(fit_range), num=4 * len(fit_range)
    )

    samples = fitresult.flatchain
    params = fitresult.params.copy()

    df = pd.DataFrame(columns=["weq", "w50p", "w10p"])

    for idx in range(len(samples)):
        for field in samples.columns:
            params[field].set(value=samples.loc[idx, field])

        amps = fitresult.eval(params=params, x=dense_range)

        weq_post = pulsemodels.equivalent_width(dense_range, amps)
        w50_post = pulsemodels.full_width_post(dense_range, amps, 0.5)
        w10_post = pulsemodels.full_width_post(dense_range, amps, 0.1)

        temp = pd.DataFrame(
            {"weq": weq_post, "w50p": w50_post, "w10p": w10_post},
            index=[idx],
        )

        df = pd.concat([df, temp], ignore_index=True)

    # convert object to numeric
    df = df.apply(pd.to_numeric)

    widths = {}

    for field in df.columns:
        quantiles = np.quantile(df[field], q=[0.16, 0.5, 0.84])

        error = np.maximum(
            np.abs(quantiles[1] - quantiles[0]), np.abs(quantiles[2] - quantiles[1])
        )

        widths[field] = {"value": quantiles[1], "error": error}

    return widths


def fit_profile(cand, plot_range, fscrunch_factor, smodel, params):
    """
    Fit an FRB profile.
    """

    df = pd.DataFrame(
        columns=[
            "band",
            "cfreq",
            "fluence",
            "err_fluence",
            "center",
            "err_center",
            "sigma",
            "err_sigma",
            "weq",
            "err_weq",
            "w50p",
            "err_w50p",
            "w10p",
            "err_w10p",
        ]
    )

    fitresults = []
    for iband in range(cand.dynspec.shape[0]):
        print("\nRunning sub-band: {0}".format(iband))

        sub_profile = cand.dynspec[iband, :]

        # select only this time range of data for the fit
        mask = (plot_range >= params["fitrange"][0]) & (
            plot_range <= params["fitrange"][1]
        )
        fit_range = np.copy(plot_range[mask])
        sub_profile = sub_profile[mask]

        # remove baseline and normalise
        sub_profile = sub_profile - np.mean(sub_profile)
        sub_profile = sub_profile / np.max(sub_profile)

        # compute baseline statistics outside the central +- 30 ms
        # XXX: we should adjust this based on the actual pulse width
        mask = np.abs(fit_range) > 30.0
        quantiles = np.quantile(sub_profile[mask], q=[0.25, 0.75], axis=None)
        std = 0.7413 * np.abs(quantiles[1] - quantiles[0])
        snr = np.max(sub_profile[~mask]) / std
        print("S/N: {0:.2f}".format(snr))

        if not snr >= params["snr"]:
            print("Profile S/N too low: {0:.2f}".format(snr))
            continue

        idx_hi = iband * fscrunch_factor
        idx_lo = idx_hi + fscrunch_factor - 1
        f_hi = cand.freqs[idx_hi]
        f_lo = cand.freqs[idx_lo]
        cfreq = 0.5 * (f_hi + f_lo)
        chan_bw = np.abs(np.diff(cand.freqs))[0]

        dm_smear = get_dm_smearing(
            cfreq - 0.5 * chan_bw, cfreq + 0.5 * chan_bw, cand.dm
        )

        print(
            "Frequencies (MHz), DM smearing (ms): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}".format(
                f_lo, cfreq, f_hi, dm_smear
            )
        )

        params["f_lo"] = f_lo
        params["f_hi"] = f_hi
        params["cfreq"] = cfreq
        params["dm_smear"] = dm_smear

        fitresult = fit_profile_model(fit_range, sub_profile, smodel, params)
        fitresults.append(fitresult)

        # fit an unscattered model for comparison
        if params["compare"]:
            fitresult2 = fit_profile_model(
                fit_range, sub_profile, "unscattered", params
            )
        else:
            fitresult2 = None

        plotting.plot_profile_fit(
            fit_range, sub_profile, fitresult, iband, cfreq, params, fitresult2
        )

        # compute profile statistics
        widths_post = compute_post_widths(fit_range, fitresult)

        temp = pd.DataFrame(
            {
                "band": iband,
                "cfreq": cfreq,
                "fluence": fitresult.best_values["fluence"],
                "err_fluence": fitresult.params["fluence"].stderr,
                "center": fitresult.best_values["center"],
                "err_center": fitresult.params["center"].stderr,
                "sigma": fitresult.best_values["sigma"],
                "err_sigma": fitresult.params["sigma"].stderr,
                "weq": widths_post["weq"]["value"],
                "err_weq": widths_post["weq"]["error"],
                "w50p": widths_post["w50p"]["value"],
                "err_w50p": widths_post["w50p"]["error"],
                "w10p": widths_post["w10p"]["value"],
                "err_w10p": widths_post["w10p"]["error"],
            },
            index=[iband],
        )

        if "taus" in fitresult.best_values:
            temp["taus"] = fitresult.best_values["taus"]
            temp["err_taus"] = fitresult.params["taus"].stderr

        if "taud" in fitresult.best_values:
            temp["taud"] = fitresult.best_values["taud"]

        df = pd.concat([df, temp], ignore_index=True)

    # convert object to numeric
    df = df.apply(pd.to_numeric)

    # compute intrinsic w50 and w10
    df["w50i"] = pulsemodels.gaussian_fwhm(df["sigma"])
    df["err_w50i"] = pulsemodels.gaussian_fwhm(df["err_sigma"])
    df["w10i"] = pulsemodels.gaussian_fwtm(df["sigma"])
    df["err_w10i"] = pulsemodels.gaussian_fwtm(df["err_sigma"])

    return df, fitresults


#
# MAIN
#


def main():
    args = parse_args()

    # sanity check command line arguments
    check_args(args)

    plotting.use_custom_matplotlib_formatting()
    # scatfit creates many diagnostic figures on purpose, silence the warning
    plt.rcParams.update({"figure.max_open_warning": 0})

    if args.show_models:
        plotting.plot_profile_models()
        plt.show()
        sys.exit(0)

    if not os.path.isfile(args.filename):
        print("The file does not exist: {0}".format(args.filename))
        sys.exit(1)

    # check if the file is filterbank or archive
    if os.path.splitext(args.filename)[1] == ".fil":
        cand = sigproc.load_frb_data(
            args.filename,
            args.dm,
            args.fscrunch_factor,
            args.tscrunch_factor,
            args.norfi,
        )
    else:
        cand = archive.load_frb_data(
            args.filename,
            args.dm,
            args.fscrunch_factor,
            args.tscrunch_factor,
            args.norfi,
        )

    # band-integrated profile
    profile = np.sum(cand.dynspec, axis=0)
    profile = profile - np.mean(profile)
    profile = profile / np.max(profile)

    fact = 1000 * cand.tsamp * args.tscrunch_factor
    plot_range = np.linspace(0, fact * len(profile), num=len(profile))

    # centre on the burst
    if args.bin_burst is not None:
        bin_burst = args.bin_burst
    else:
        bin_burst = np.argmax(profile)
    plot_range -= fact * bin_burst

    params = {
        "compare": args.compare,
        "fast": args.fast,
        "fitrange": args.fitrange,
        "publish": args.publish,
        "snr": args.snr,
        "zoom": args.zoom,
    }

    # fit integrated profile
    fit_df, fit_results = fit_profile(
        cand, plot_range, args.fscrunch_factor, args.smodel, params
    )
    print("\nFit results")
    print(fit_df)

    # Saving resut
    fnout = "scattering_fit_result.csv"
    fit_df.to_csv(fnout)

    # compute updated dm
    if len(fit_df.index) >= 2:
        plotting.plot_center_scaling(fit_df)
        compute_updated_dm(fit_df, args.dm, params)

    if args.smodel == "unscattered" and args.tscrunch_factor == 1:
        # best topocentric burst arrival time
        # at the highest frequency channel
        try:
            start_mjd = Time(
                cand._header["tstart"], format="mjd", scale="utc", precision=9
            )
        except AttributeError:
            start_mjd = Time(cand.tstart, format="mjd", scale="utc", precision=9)
        burst_offset = TimeDelta(
            bin_burst * cand.tsamp * args.tscrunch_factor, format="sec"
        )
        fit_offset = TimeDelta(1.0e-3 * fit_df["center"].iloc[0], format="sec")
        mjd_topo = start_mjd + burst_offset + fit_offset
        print(
            "Topocentric burst arrival time at {0} MHz: MJD {1}".format(
                cand.fch1, mjd_topo
            )
        )

    if args.fit_scatindex and len(fit_df.index) >= 2 and "taus" in fit_df.columns:
        fitresult = fit_powerlaw(
            1e-3 * fit_df["cfreq"].to_numpy(),
            fit_df["taus"].to_numpy(),
            fit_df["err_taus"].to_numpy(),
            params,
        )
    else:
        fitresult = None

    plotting.plot_width_scaling(fit_df, cand, fitresult)

    plotting.plot_frb(cand, plot_range, profile, params)

    plotting.plot_frb_scat(
        cand, fit_df, fit_results, args.smodel, plot_range, params, dynspec=True
    )

    plotting.plot_frb_scat(
        cand, fit_df, fit_results, args.smodel, plot_range, params, dynspec=False
    )

    plt.show()

    print("All done.")


if __name__ == "__main__":
    main()
