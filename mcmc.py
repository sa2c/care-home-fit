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

STEP_METHODS = {
    'slice': pm.Slice,
    'metropolis': pm.Metropolis
}


class CareHomeLikelihood(tt.Op):
    """
    A class to wrap the log-likelihood function defined in likelihood.py
    so that it can be used with pymc3.
    """

    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, cases_file, covariates_file,
                 dist_params=None, discharges_file=None,
                 fixed_r_c=None, fixed_r_h=None):
        """
        Initialise the Op with the data required by the log-likelihood function.
        """

        self.fixed_r_c = fixed_r_c
        self.fixed_r_h = fixed_r_h
        self.dist_params = dist_params
        _, self.cases, self.covariates, self.discharges = (
            likelihood.safely_read_cases_covariates_discharges(
                cases_file, covariates_file, discharges_file
            )
        )
        if (self.fixed_r_h != 0) and (self.discharges is None):
            raise ValueError(
                "fixed_r_h set to True, but not discharge data found"
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

        start_index = 0
        if self.fixed_r_c is None:
            r_c = inputs[0][0]
            start_index += 1
        else:
            r_c = self.fixed_r_c

        if self.fixed_r_h is None:
            r_h = inputs[0][start_index]
            start_index += 1
        else:
            r_h = self.fixed_r_h

        fit_params = {
            'baseline_intensities': np.asarray(
                inputs[0][start_index:]
            ),
            'r_c': r_c,
            'r_h': r_h
        }

        if (r_c == 0) and (r_h == 0):
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
              fixed_r_c=None, fixed_r_h=None, sigmas=None,
              num_baseline_intensities=4):
    """
    Construct a pymc3 Model.
    Parameters:
     - log_likelihood: an instance of CareHomeLikelihood
     - fixed_r_c, fixed_r_h: a number or None. If a number, then the value is
       fixed (and if zero, the calculation may be skipped). If None, then
       the value is fitted.
     - sigmas: an optional list of values of sigma for the HalfNormal prior
       distributions on the fit parameters; default is all 1.0
     - n_bi: the number of baseline intensities (i.e. care home sizes)
    Returns:
     - A pymc3 Model object.
    """
    num_params = num_baseline_intensities + (fixed_r_c != 0) + (fixed_r_h != 0)
    if sigmas is None:
        sigmas = [1.0] * num_params
    params = []
    with pm.Model() as model:
        if fixed_r_c is None:
            r_c = pm.HalfNormal('r_c', sigma=sigmas.pop(0))
            params.append(r_c)

        if fixed_r_h is None:
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


def mcmc_fit(model, num_draws=100, num_burn=100, step='slice'):
    """
    Run the supplied model and return the trace.
    Parameters:
     - model: a pymc3 Model instance
     - num_draws, num_burn: the number of iterations of the Monte Carlo to run
       as thermalisation and as production.
     - step: the step method to use
    Returns:
     - A pymc3 trace object.
    """

    if step not in STEP_METHODS:
        raise ValueError(f"Supplied method {step} is not valid.")

    with model:
        trace = pm.sample(
            num_draws,
            tune=num_burn,
            discard_tuned_samples=True,
            step=STEP_METHODS[step]()
        )

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

    def get_with_text(fixed_r):
        if fixed_r == 0:
            return 'omitted'
        elif fixed_r is None:
            return 'fitted'
        else:
            return f'fixed at {fixed_r}'

    with open(filename, 'w') as file_handle:
        print_to_stdout_and_file(
            f"Fit with self-excitation "
            f"{get_with_text(likelihood_obj.fixed_r_c)} and "
            f"discharge excitation {get_with_text(likelihood_obj.fixed_r_h)}:",
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
        fixed_r_c=None,
        fixed_r_h=None,
        discharges_filename=None,
        output_prefix='',
        step='slice'
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
        fixed_r_c=fixed_r_c, fixed_r_h=fixed_r_h
    )
    model = get_model(
        likelihood_obj,
        fixed_r_c=fixed_r_c,
        fixed_r_h=fixed_r_h,
        num_baseline_intensities=num_baseline_intensities
    )
    trace = mcmc_fit(model, num_draws=num_draws, num_burn=num_burn, step=step)
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
        {'default': 2000, 'type': int},
    ), (
        ['--num_burn'],
        {'default': 500, 'type': int}
    ), (
        ['--output_directory'],
        {'required': True}
    ), (
        ['--overwrite'],
        {'action': 'store_true'}
    ), (
        ['--case'],
        {'default': None}
    ), (
        ['--step'],
        {'default': 'slice'}
    )]
    args, fit_params, dist_params = likelihood.get_params_from_args(extra_args)
    if len(fit_params['baseline_intensities']) == 1:
        num_baseline_intensities = int(fit_params['baseline_intensities'])
    else:
        num_baseline_intensities = len(fit_params['baseline_intensities'])

    case_options = {
        "base": (0, 0),
        "self": (fit_params['r_c'], 0),
        "full": (fit_params['r_c'], fit_params['r_h'])
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

    for output_prefix, fixed_r_c, fixed_r_h in cases:
        create_and_run_model(
            cases_filename=args.cases_file,
            covariates_filename=args.covariates_file,
            dist_params=dist_params,
            num_baseline_intensities=num_baseline_intensities,
            num_draws=args.num_draws,
            num_burn=args.num_burn,
            fixed_r_c=fixed_r_c,
            fixed_r_h=fixed_r_h,
            output_prefix=output_directory / output_prefix,
            discharges_filename=args.discharges_file,
            step=args.step
        )


if __name__ == '__main__':
    main()
