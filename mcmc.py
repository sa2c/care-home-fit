#!/usr/bin/env python

"""
Allow fitting of care home data via the functions in likelihood
using pymc3.
"""


import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt

import likelihood


class CareHomeLikelihood(tt.Op):
    """
    A class to wrap the log-likelihood function defined in likelihood.py
    so that it can be used with pymc3.
    """

    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, cases_file, covariates_file,
                 dist_params=None, discharges_file=None,
                 with_r_c=True, with_r_h=True):
        """
        Initialise the Op with the data required by the log-likelihood function.
        """

        self.with_r_c = with_r_c
        self.with_r_h = with_r_h
        self.dist_params = dist_params
        _, self.cases, self.covariates, self.discharges = (
            likelihood.safely_read_cases_covariates_discharges(
                cases_file, covariates_file, discharges_file
            )
        )
        if self.with_r_h and (self.discharges is None):
            raise ValueError(
                "with_r_h set to True, but not discharge data found"
            )

    def __call__(self, v):
        """
        Allow instances to be called via a dictionary, as pymc3 requires.
        """

        # Documentation suggests using a lambda for this, but this prevents
        # pickling from working.

        return super().__call__(v)

    def perform(self, node, inputs, outputs):
        """
        Perform the Op; get the log-likelihood of the data given the inputs.
        """

        if self.with_r_c:
            r_c = inputs[0][0]
        else:
            r_c = 0

        if self.with_r_h:
            r_h = inputs[0][0 + self.with_r_c]
        else:
            r_h = None

        fit_params = {
            'baseline_intensities': np.asarray(
                inputs[0][self.with_r_c + self.with_r_h:]
            ),
            'r_c': r_c,
            'r_h': r_h
        }

        if (r_c == 0) and (r_h is None):
            intensity = likelihood.carehome_intensity_null(
                covariates=self.covariates,
                cases=self.cases,
                fit_params=fit_params
            )
        else:
            intensity = likelihood.carehome_intensity(
                covariates=self.covariates,
                cases=self.cases,
                discharges=self.discharges,
                fit_params=fit_params,
                dist_params=self.dist_params
            )

        logl = likelihood.likelihood(intensity, self.cases)
        if outputs is not None:
            outputs[0][0] = np.array(logl)
        else:
            return logl


def get_model(log_likelihood,
              with_r_c=True, with_r_h=True, sigmas=None,
              num_baseline_intensities=4):
    """
    Construct a pymc3 Model.
    Parameters:
     - log_likelihood: an instance of CareHomeLikelihood
     - with_r_c, with_r_h: whether the self- and discharge excitation
       parameters should be included in the model
     - sigmas: an optional list of values of sigma for the HalfNormal prior
       distributions on the fit parameters; default is all 1.0
     - n_bi: the number of baseline intensities (i.e. care home sizes)
    Returns:
     - A pymc3 Model object.
    """
    num_params = num_baseline_intensities + with_r_c + with_r_h
    if sigmas is None:
        sigmas = [1.0] * num_params
    params = []
    with pm.Model() as model:
        if with_r_c:
            r_c = pm.HalfNormal('r_c', sigma=sigmas.pop(0))
            params.append(r_c)

        if with_r_h:
            r_h = pm.HalfNormal('r_h', sigma=sigmas.pop(0))
            params.append(r_h)

        bi = []
        for bi_index in range(num_baseline_intensities):
            bi.append(pm.HalfNormal(f'bi_{bi_index}', sigma=sigmas.pop(0)))
        params.extend(bi)

        params_tensor = tt.as_tensor_variable(params)

        ll = pm.DensityDist(
            'likelihood',
            log_likelihood,
            observed={'v': params_tensor}
        )

    return model


def mcmc_fit(model, num_draws=100, num_burn=100):
    """
    Run the supplied model and return the trace.
    Parameters:
     - model: a pymc3 Model instance
     - num_draws, num_burn: the number of iterations of the Monte Carlo to run
       as thermalisation and as production.
    Returns:
     - A pymc3 trace object.
    """

    with model:
        trace = pm.sample(num_draws, tune=num_burn, discard_tuned_samples=True)

    return trace


def plot_trace(trace, filename):
    """
    Plot the given trace, saving the result as the specified filename.
    """

    pm.traceplot(trace)

    fig = plt.gcf()
    fig.savefig(filename)
    plt.close(fig)


def print_to_stdout_and_file(
        thing_to_print, to_stdout=True, file_handle=None
):
    """
    Takes the supplied thing_to_print, and prints it both to stdout
    (unless to_stdout is False) and to the file pointed at by file_handle
    (if any).
    """
    if file_handle:
        print(thing_to_print, file=file_handle)
    if to_stdout:
        print(thing_to_print)


def print_result(summary, likelihood_obj, filename='/dev/null'):
    """
    Given a summary, output it, then calculate and output the likelihood
    associated with these fit parameters.
    """

    with_text = {True: 'with', False: 'without'}
    with open(filename, 'w') as file_handle:
        print_to_stdout_and_file(
            f"Fit {with_text[likelihood_obj.with_r_c]} self-excitation and "
            f"{with_text[likelihood_obj.with_r_h]} discharge excitation:",
            file_handle=file_handle
        )
        print_to_stdout_and_file(summary, file_handle=file_handle)
        likelihood_result = likelihood_obj.perform(
            None,
            [list(summary['mean'])],
            None
        )
        print_to_stdout_and_file(
            f"Likelihood at mean parameter values is {likelihood_result}.",
            file_handle=file_handle
        )
    print()


def create_and_run_model(
        cases_filename,
        covariates_filename,
        dist_params,
        num_baseline_intensities,
        num_draws=100,
        num_burn=100,
        with_r_c=True,
        with_r_h=True,
        discharges_filename=None,
        output_prefix=''
):
    """
    Perform a fit of a given set of cases, covariates, and optionally
    discharges, with specified distribution parameters, and using the
    likelihood form defined in likelihood.py. Output the MCMC history to
    disk. Output a plot of the history to disk. Output a summary of the fitted
    parameters to the screen and to disk.
    """

    likelihood_obj = CareHomeLikelihood(
        cases_filename, covariates_filename, dist_params, discharges_filename,
        with_r_c=with_r_c, with_r_h=with_r_h
    )
    model = get_model(
        likelihood_obj,
        with_r_c=with_r_c,
        with_r_h=with_r_h,
        num_baseline_intensities=num_baseline_intensities
    )
    trace = mcmc_fit(model, num_draws=num_draws, num_burn=num_burn)
    plot_trace(
        trace,
        output_prefix.parent / (output_prefix.name + 'traceplot.pdf')
    )
    pm.save_trace(
        trace,
        output_prefix.parent / (output_prefix.name + 'trace.dat')
    )
    summary = pm.summary(trace, round_to="none")
    print_result(
        summary,
        likelihood_obj,
        filename=output_prefix.parent / (output_prefix.name + 'summary.txt')
    )


def get_output_directory(directory_name, overwrite=False):
    '''
    Create a directory with name directory_name. Fail if the directory already
    exists, unless overwrite is set to True, in which case delete the existing
    directory or file of the same name.
    '''

    from pathlib import Path
    from shutil import rmtree

    output_directory = Path(directory_name)
    if output_directory.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory {output_directory} already exists; please "
            "choose another one or try again with '--overwrite'"
        )
    if output_directory.exists():
        rmtree(output_directory)
    output_directory.mkdir()
    return output_directory


def main():
    """
    Parse command-line options and run the three cases of interest.
    """

    extra_args = [(
        ['--num_draws'],
        {'default': 100, 'type': int},
    ), (
        ['--num_burn'],
        {'default': 100, 'type': int}
    ), (
        ['--output_directory'],
        {'required': True}
    ), (
        ['--overwrite'],
        {'action': 'store_true'}
    ), (
        ['--case'],
        {'default': None}
    )]
    args, fit_params, dist_params = likelihood.get_params_from_args(extra_args)
    if len(fit_params['baseline_intensities']) == 1:
        num_baseline_intensities = int(fit_params['baseline_intensities'])
    else:
        num_baseline_intensities = len(fit_params['baseline_intensities'])

    case_options = {
        "base": (False, False),
        "self": (True, False),
        "full": (True, True)
    }
    get_case = lambda case : (case, *case_options[case])
    if args.case is not None:
        cases = [get_case(args.case)]
    else:
        cases = [get_case('base'), get_case('self')]
        if args.discharges_file:
            cases.append(get_case('full'))

    output_directory = get_output_directory(
        args.output_directory,
        args.overwrite
    )

    for output_prefix, with_r_c, with_r_h in cases:
        create_and_run_model(
            cases_filename=args.cases_file,
            covariates_filename=args.covariates_file,
            dist_params=dist_params,
            num_baseline_intensities=num_baseline_intensities,
            num_draws=args.num_draws,
            num_burn=args.num_burn,
            with_r_c=with_r_c,
            with_r_h=with_r_h,
            output_prefix=output_directory / output_prefix,
            discharges_filename=args.discharges_file
        )


if __name__ == '__main__':
    main()
