#!/usr/bin/env python

'''
Functions to calculate the predicted intensity of cases at carehomes based on
the care home size, number of cases previously seen there, and number of
hospital discharges to the carehome, and to calculate the likelihood of that
scenario.'''


from numpy import asarray, loadtxt, equal, zeros_like, arange, log, newaxis
from numpy import sum as npsum
from scipy.special import gammaln
from scipy.stats import gamma


DEFAULT_DIST_PARAMS = {
    'self_excitation_mean': 6.5,
    'self_excitation_cv': 0.62,
    'discharge_excitation_mean': 6.5,
    'discharge_excitation_cv': 0.62
}


def compactify(array):
    '''Take an array of uint32s and returns it in the smallest datatype.'''
    if array.max() < 256:
        compact_array = array.astype('uint8')
    elif array.max() < 65536:
        compact_array = array.astype('uint16')
    else:
        compact_array = array
    return compact_array


def read_and_tidy_data(filename):
    '''Read in the CSV file in `filename`, sort by the values in the first row
    (i.e. ensure that data are sorted by care homes), then return separately
    the first row (the ID list) and the subsequent data in the smallest data
    type into which it will fit.

    Assumes all values are integers smaller than 2**32'''

    read_data = loadtxt(filename, dtype="uint32", delimiter=',')
    sorted_read_data = read_data[:, read_data[0, :].argsort()]
    care_home_ids = compactify(sorted_read_data[0, :])
    values = compactify(sorted_read_data[1:, :])
    return care_home_ids, values


def carehome_intensity_null(covariates, cases, fit_params, **kwargs):
    '''
    Returns the intensity of cases at all care homes on all dates, assuming no
    effect due to cases or discharges.
    Shape is (N_DATES, N_CARE_HOMES).
    Requires:
    Input data:
     - covariates: a (1, N_CARE_HOMES) array of care home size bands
     - cases: an (N_DATES, N_CARE_HOMES) array of integers representing
              the number of cases at a given care home on a given day
    fit_parameters: a dict containing elements:
     - baseline_intensities: a 1d array of baseline intensities as a function
                             of care home size band
    '''
    return (zeros_like(cases, dtype="float64")
            + fit_params['baseline_intensities'][covariates])


def single_excitation(triggers, shape, scale):
    '''
    Calculates a single set of excitation terms in the form
        e_i(t) = \\sum_{s<t} f(t - s) triggers_i(s)
    where f is the gamma pdf with given shape and scale,
    and triggers is a 2-d array indexed as (t, i)'''

    n_dates, _ = triggers.shape
    date_delta_range = arange(1, n_dates)[:, newaxis]

    # Generate an array of all values from the distribution to be used
    # given the range of date differences to be considered
    # This avoids recalculating on each loop iteration
    full_f_c = gamma.pdf(date_delta_range, a=shape, scale=scale)
    output = zeros_like(triggers, dtype='float64')

    for term in range(1, n_dates):
        # On each loop iteration, consider terms arising from one date
        # This is the reverse of the form of the summation expressed
        # analytically, but maps more nicely onto the data structures
        # and how numpy likes to deal with them
        output[term:] += full_f_c[:n_dates - term] * triggers[term - 1]
    return output


def carehome_intensity(fit_params, covariates, cases, dist_params,
                       discharges=None):
    '''
    Returns the intensity of cases at all care homes on all dates.
    Shape is (N_DATES, N_CARE_HOMES).
    Requires:
    Input data:
     - covariates: a (1, N_CARE_HOMES) array of care home size bands
     - cases: an (N_DATES, N_CARE_HOMES) array of integers representing
              the number of cases at a given care home on a given day
     - discharges: an (N_DATES, N_CARE_HOMES array of integers representing
              the number of discharges from hospital into a given care home
              on a given day
    fit_params - a dict containing elements:
     - baseline_intensities: a 1d array of baseline intensities as a function
                             of care home size band
     - r_c: self-excitation
     - r_h: excitation by hospital. If zero, then calculation of this term is
       skipped
    dist_params - a dict containing elements
     - self_excitation_shape, self_excitation_scale - parameters for the gamma
       distribution of f_c
     - discharge_excitation_shape, discharge_excitation_scale - parameters for
       the gamma distribution of f_h
    '''

    output = carehome_intensity_null(
        covariates=covariates,
        cases=cases,
        fit_params=fit_params
    )
    output += fit_params['r_c'] * single_excitation(
        cases,
        dist_params['self_excitation_shape'],
        dist_params['self_excitation_scale']
    )

    if fit_params.get('r_h') is not None:
        assert discharges is not None
        output += fit_params['r_h'] * single_excitation(
            discharges,
            dist_params['discharge_excitation_shape'],
            dist_params['discharge_excitation_scale']
        )

    return output


def calculate_gamma_parameters(mean, cv):
    '''Takes in mean and coefficient of variation cv of a gamma distribution,
    and outputs the shape and scale parameters required by Scipy.'''

    return cv ** -2, mean * cv ** 2


def likelihood(intensity, cases):
    '''
    Sum over care homes i and dates t:
    ln(\\lambda^k exp(-\\lambda) / (k!))
      = (k ln \\lambda - \\lambda - ln(k!))
    where k = cases and \\lambda = intensity
    '''
    non_zero_cases = (cases > 0)

    # gammaln(n) = ln((n-1)!) for integer n
    return (npsum(cases[non_zero_cases] * log(intensity[non_zero_cases]))
            - npsum(intensity + gammaln(cases + 1)))


def safely_read_cases_covariates_discharges(
        cases_file, covariates_file, discharges_file=None
):
    '''Read in cases, covariates, and optionally discharges from specified
    CSV files and check their consistency before returning them.'''

    care_home_ids, cases = read_and_tidy_data(cases_file)
    covariate_ch_ids, covariates = read_and_tidy_data(covariates_file)
    assert equal(care_home_ids, covariate_ch_ids).all()

    if discharges_file is not None:
        discharge_ch_ids, discharges = read_and_tidy_data(discharges_file)
        assert equal(care_home_ids, discharge_ch_ids).all()
    else:
        discharges = None

    return care_home_ids, cases, covariates, discharges


def calculate_likelihood_from_files(cases_file, covariates_file,
                                    discharges_file=None, **kwargs):
    '''Does the end-to-end calculation of the likelihood from files for cases,
    covariates, and discharges stored on disk.'''

    _, cases, covariates, discharges = safely_read_cases_covariates_discharges(
        cases_file, covariates_file, discharges_file
    )
    if (
            not kwargs['fit_params'].get('r_c')
            and not kwargs['fit_params'].get('r_h')
    ):
        return likelihood(
            carehome_intensity_null(covariates=covariates,
                                    cases=cases,
                                    **kwargs),
            cases
        )
    else:
        return likelihood(
            carehome_intensity(
                covariates=covariates,
                cases=cases,
                discharges=discharges,
                **kwargs
            ),
            cases
        )


def get_fittable_likelihood(cases_file, covariates_file, discharges_file=None):
    '''Closure that returns a function wrapping likelihood and intensity such
    that likelihood can be maximised by scipy.optimize.'''

    _, cases, covariates, discharges = safely_read_cases_covariates_discharges(
        cases_file, covariates_file, discharges_file
    )

    def fittable_likelihood(fit_params, *dist_params):
        '''Calculate the likelihood of the enclosed data for the specified
        set of fit and distribution parameters:
         - fit_params: a 1d-array with elements:
           * 0 is the case excitation coefficient r_c
           * 1 is the discharge excitation coefficient r_h
           * 2: are the baseline intensities
         - dist_parameters: a tuple of the four elements self_excitation_shape,
           self_excitation_scale, discharge_excitation_shape,
           discharge_excitation_scale parametrising the relevant
           gamma pdfs.'''

        r_c, r_h = fit_params[:2]
        baseline_intensities = fit_params[2:]
        fit_params_dict = {
            'baseline_intensities': baseline_intensities,
            'r_c': r_c,
            'r_h': r_h
        }
        dist_params_dict = dict(zip(
            ('self_excitation_shape', 'self_excitation_scale',
             'discharge_excitation_shape', 'discharge_excitation_scale'),
            dist_params
        ))

        if (r_c == 0) and (r_h == 0):
            intensity = carehome_intensity_null(
                covariates=covariates,
                cases=cases,
                fit_params=fit_params_dict,
                dist_params=dist_params_dict
            )
        else:
            intensity = carehome_intensity(
                covariates=covariates,
                cases=cases,
                discharges=discharges,
                fit_params=fit_params_dict,
                dist_params=dist_params_dict
            )

        return -likelihood(intensity, cases)

    return fittable_likelihood


def get_params_from_args():
    '''Parse command-line arguments to get filenames for cases,
    covariates, and discharges, and to define any fit and distribution
    parameters.'''

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--cases_file', required=True)
    parser.add_argument('--covariates_file', required=True)
    parser.add_argument('--discharges_file', default=None)
    parser.add_argument('--baseline_intensities',
                        required=True, nargs='+', type=float)
    parser.add_argument('--r_c', default=None, type=float)
    parser.add_argument('--r_h', default=None, type=float)

    for param_name, param_value in DEFAULT_DIST_PARAMS.items():
        parser.add_argument(f'--{param_name}', type=float, default=param_value)

    args = parser.parse_args()

    fit_params = {
        'baseline_intensities': asarray(args.baseline_intensities),
        'r_c': args.r_c,
        'r_h': args.r_h
    }

    dist_params = {}
    (
        dist_params['self_excitation_shape'],
        dist_params['self_excitation_scale']
    ) = calculate_gamma_parameters(
        args.self_excitation_mean,
        args.self_excitation_cv
    )

    (
        dist_params['discharge_excitation_shape'],
        dist_params['discharge_excitation_scale']
    ) = calculate_gamma_parameters(
        args.discharge_excitation_mean,
        args.discharge_excitation_cv
    )

    return args, fit_params, dist_params


def main():
    '''Perform a quick test of the intensity calculation.'''

    from pprint import pprint
    from time import time

    args, fit_params, dist_params = get_params_from_args()

    print("Calculating likelihood with fit parameters:")
    pprint(fit_params)
    print("and distribution parameters:")
    pprint(dist_params)
    print()

    start_time = time()
    result = calculate_likelihood_from_files(
        cases_file=args.cases_file,
        covariates_file=args.covariates_file,
        discharges_file=args.discharges_file,
        dist_params=dist_params,
        fit_params=fit_params
    )
    end_time = time()

    print(f"Likelihood: {result}")
    print(f"Time elapsed: {end_time -  start_time}s")


if __name__ == '__main__':
    main()
