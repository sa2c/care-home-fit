#!/usr/bin/env python

'''Tests for the likelihood.py module'''


import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from scipy.stats import gamma
import likelihood


SMALL_FIT_PARAMS = {
    'baseline_intensities': np.asarray([1, 2, np.nan, np.nan]),
    'r_h': 1.5,
    'r_c': 0.5
}

SIMPLE_DIST_PARAMS = {
    'self_excitation_shape': 2,
    'self_excitation_scale': 1,
    'discharge_excitation_shape': 3,
    'discharge_excitation_scale': 2
}

SMALL_CASES_FILE = 'tests/fixtures/small.csv'
SMALL_COVARIATES_FILE = 'tests/fixtures/small_covariates.csv'

LARGE_FIT_PARAMS = {
    'baseline_intensities': np.asarray([0.3, 0.4, 0.6, 0.9]),
    'r_h': 1.5,
    'r_c': 0.5
}

FULL_DIST_PARAMS = {
    'self_excitation_shape': 2.6,
    'self_excitation_scale': 2.5,
    'discharge_excitation_shape': 2.6,
    'discharge_excitation_scale': 2.5
}


def test_gamma_pdf():
    x = np.linspace(0, 10, 100)
    shape = FULL_DIST_PARAMS['self_excitation_shape']
    scale = FULL_DIST_PARAMS['self_excitation_scale']
    assert_almost_equal(
        gamma.pdf(x, a=shape, scale=scale),
        likelihood.gamma_pdf(x, shape, scale)
    )


@pytest.mark.parametrize(
    "test_element,result_dtype",
    [(123_456_789, np.uint32), (65_535, np.uint16), (255, np.uint8)]
)
def test_compactify(test_element, result_dtype):
    '''Test that arrays compactify correctly, and to the correct data types'''
    array = np.asarray([[1, 2], [3, test_element]], dtype=np.uint32)
    result = likelihood.compactify(array)
    assert result.dtype == result_dtype
    assert_array_equal(array, result)


def test_read_and_tidy_data():
    '''Test that a CSV file with care home IDs as a header row
    is read, sorted, and split correctly.'''
    ids, values = likelihood.read_and_tidy_data(SMALL_CASES_FILE)
    assert_array_equal(ids, [14, 16, 35])
    assert_array_equal(
        values,
        [[4, 1, 6], [4, 0, 3], [6, 66, 2]]
    )

@pytest.fixture
def small_cases():
    '''Get a small data file that could be cases or discharges.'''
    return likelihood.read_and_tidy_data(SMALL_CASES_FILE)


@pytest.fixture
def small_covariates():
    '''Get a small data file containing covariates.'''
    return likelihood.read_and_tidy_data(SMALL_COVARIATES_FILE)


def test_carehome_intensity_null(small_cases, small_covariates):
    '''Test that calculating the null-case intensity (based on mapping banded
    carehome size to a base intensity) gives the correct result'''
    _, cases = small_cases
    _, covariates = small_covariates
    intensity = likelihood.carehome_intensity_null(
        covariates=covariates,
        cases=cases,
        fit_params=SMALL_FIT_PARAMS
    )
    assert_array_equal(intensity, [[1, 2, 2], [1, 2, 2], [1, 2, 2]])


def test_single_excitation(small_cases, small_covariates):
    '''Test that excitation terms of the form
        e_i(t) = \\sum_{s<t} f(t - s) triggers_i(s)
    are correctly calculated'''
    _, cases = small_cases
    _, covariates = small_covariates
    excitation = likelihood.single_excitation(cases, 2, 1)
    assert_almost_equal(
        excitation,
        [[0, 0, 0], [1.472, 0.368, 2.207], [2.554, 0.271, 2.728]],
        decimal=3
    )


def test_carehome_intensity_no_discharges(small_cases, small_covariates):
    '''Test that the behaviour of carehome_intensity in the case where
    discharges are not considered.'''
    _, cases = small_cases
    _, covariates = small_covariates
    fit_params_no_rh = {**SMALL_FIT_PARAMS, 'r_h': None}
    intensity = likelihood.carehome_intensity(
        covariates=covariates,
        cases=cases,
        fit_params=fit_params_no_rh,
        dist_params=SIMPLE_DIST_PARAMS
    )
    assert_almost_equal(
        intensity,
        [[1, 2, 2], [1.736, 2.184, 3.104], [2.277, 2.135, 3.364]],
        decimal=3
    )


def test_carehome_intensity_with_discharges(small_cases, small_covariates):
    '''Test that the behaviour of carehome_intensity is correct in the case
    where discharges are considered.'''
    _, cases = small_cases
    _, covariates = small_covariates
    discharges = cases[::-1]
    intensity = likelihood.carehome_intensity(
        covariates=covariates,
        cases=cases,
        fit_params=SMALL_FIT_PARAMS,
        dist_params=SIMPLE_DIST_PARAMS,
        discharges=discharges
    )
    assert_almost_equal(
        intensity,
        [[1, 2, 2], [2.077, 5.937, 3.217], [3.332, 11.240, 3.810]],
        decimal=3
    )


@pytest.mark.parametrize("mean, cv, expected_shape, expected_scale",
                         [(1, 1, 1, 1), (6.5, 0.62, 2.601, 2.499)])
def test_calculate_gamma_parameters(mean, cv, expected_shape, expected_scale):
    '''Test that calculation of Scipy-style gamma parameters from "descriptive"
    gamma parameters is correct.'''
    shape, scale = likelihood.calculate_gamma_parameters(mean, cv)
    assert_almost_equal([shape, scale], [expected_shape, expected_scale],
                        decimal=3)


def test_likelihood():
    '''Test that the likelihood calculation is correct'''

    cases = np.asarray([[3, 1, 0, 1], [1, 0, 2, 1], [0, 0, 0, 1]])
    intensity = np.asarray(
        [[1, 3, 1.5, 6], [4.2, 3.1, 7, 1.4], [2, 5.1, 4.2, 8.9]]
    )

    result = likelihood.likelihood(intensity, cases)
    assert_almost_equal(result, -39.145, decimal=3)


def test_calculate_likelihood_from_files_no_discharges():
    '''Test that likelihood is correctly calculated from input files
    when discharges are not considered.'''
    fit_params_no_rh = {**SMALL_FIT_PARAMS, 'r_h': None}
    result = likelihood.calculate_likelihood_from_files(
        SMALL_CASES_FILE, SMALL_COVARIATES_FILE,
        fit_params=fit_params_no_rh, dist_params=SIMPLE_DIST_PARAMS
    )
    assert_almost_equal(result, -187.443, decimal=3)


def test_calculate_likelihood_from_files_no_cases():
    '''Test that likelihood is correctly calculated from input files
    when cases are not considered.'''
    fit_params_no_rh = {**SMALL_FIT_PARAMS, 'r_c': 0}
    result = likelihood.calculate_likelihood_from_files(
        SMALL_CASES_FILE, SMALL_COVARIATES_FILE,
        discharges_file=SMALL_CASES_FILE,
        fit_params=fit_params_no_rh, dist_params=SIMPLE_DIST_PARAMS
    )
    assert_almost_equal(result, -189.046, decimal=3)


def test_calculate_likelihood_from_files_no_discharges_or_cases():
    '''Test that likelihood is correctly calculated from input files
    when neither cases nor discharges are considered.'''
    fit_params_no_rh = {**SMALL_FIT_PARAMS, 'r_h': None, 'r_c': 0}
    result = likelihood.calculate_likelihood_from_files(
        SMALL_CASES_FILE, SMALL_COVARIATES_FILE,
        fit_params=fit_params_no_rh, dist_params=SIMPLE_DIST_PARAMS
    )
    assert_almost_equal(result, -196.466, decimal=3)


def test_calculate_likelihood_from_files_with_discharges():
    '''Test that likelihood is correctly calculated from input files
    when discharges are considered.'''
    result = likelihood.calculate_likelihood_from_files(
        SMALL_CASES_FILE, SMALL_COVARIATES_FILE,
        discharges_file=SMALL_CASES_FILE,
        fit_params=SMALL_FIT_PARAMS, dist_params=SIMPLE_DIST_PARAMS
    )
    assert_almost_equal(result, -182.761, decimal=3)


def test_calculate_likelihood_from_files_missing_discharges():
    '''Test that an error is generated when r_h is provided but discharge data
    are not'''

    with pytest.raises(AssertionError):
        likelihood.calculate_likelihood_from_files(
            SMALL_CASES_FILE, SMALL_COVARIATES_FILE,
            fit_params=SMALL_FIT_PARAMS, dist_params=SIMPLE_DIST_PARAMS
        )


@pytest.mark.parametrize(
    'r_c, r_h, expect',
    [(0, 0, 196.466),
     (0.5, 1.5, 182.761),
     (0.5, 0, 187.443),
     (0, 1.5, 189.046)]
)
def test_fittable_likelihood(r_c, r_h, expect):
    '''Test that the closure to give a version of intensity and likelihood that
    can be fitted by scipy works correctly.'''

    fittable_likelihood = likelihood.get_fittable_likelihood(
        SMALL_CASES_FILE, SMALL_COVARIATES_FILE, SMALL_CASES_FILE
    )
    fit_params = np.asarray(
        [r_c, r_h, *SMALL_FIT_PARAMS['baseline_intensities']]
    )
    assert_almost_equal(
        fittable_likelihood(
            fit_params, *map(
                SIMPLE_DIST_PARAMS.get,
                (('self_excitation_shape', 'self_excitation_scale',
                  'discharge_excitation_shape', 'discharge_excitation_scale'))
            )
        ),
        expect,
        decimal=3
    )


@pytest.fixture
def large_test_data():
    '''Generate test data of the size expected from SAIL.'''
    max_categories = 4

    num_care_homes = 1000
    num_cases = 2000
    num_case_homes = 330
    num_discharges = 3000
    num_discharge_homes = 500
    num_days = 181
    num_covariates = 1
    max_carehome_id = 32767

    cases = np.zeros((num_days, num_care_homes), dtype=np.int8)
    discharges = np.zeros((num_days, num_care_homes), dtype=np.int8)
    covariates = np.zeros((num_covariates, num_care_homes), dtype=np.int8)

    # For runs with the same version of numpy, we should get the same
    # test data each time. Not guaranteed to work between versions
    # because default_rng can change.
    rng = np.random.default_rng(seed=0)
    care_home_ids = rng.choice(
        max_carehome_id, size=num_care_homes, replace=False
    )

    for sample_array, num_instances, num_places in (
            (cases, num_cases, num_case_homes),
            (discharges, num_discharges, num_discharge_homes)
    ):
        for _ in range(num_instances):
            sample_array[rng.integers(num_days), rng.integers(num_places)] += 1

    covariates[0] = rng.choice(max_categories, size=num_care_homes)

    return care_home_ids, cases, covariates, discharges


def test_intensity_performance_base(large_test_data, benchmark):
    '''
    Test the performance of the intensity function for the base case
    '''

    _, cases, covariates, _ = large_test_data

    kwargs = {
        'fit_params': {**LARGE_FIT_PARAMS, 'r_h': None, 'r_c': None},
        'covariates': covariates,
        'cases': cases
    }
    # Ensure that numba can jit the function before timing it
    likelihood.carehome_intensity_null(**kwargs)
    benchmark(likelihood.carehome_intensity_null, **kwargs)


def test_intensity_performance_self(large_test_data, benchmark):
    '''
    Test the performance of the intensity function with self-excitation
    '''

    _, cases, covariates, _ = large_test_data

    kwargs = {
        'fit_params': {**LARGE_FIT_PARAMS, 'r_h': None},
        'covariates': covariates,
        'cases': cases,
        'dist_params': FULL_DIST_PARAMS
    }
    # Ensure that numba can jit the function before timing it
    likelihood.carehome_intensity(**kwargs)
    benchmark(likelihood.carehome_intensity, **kwargs)


def test_intensity_performance_hospitals(large_test_data, benchmark):
    '''
    Test the performance of the intensity function with self- and
    discharge excitations.'''

    _, cases, covariates, discharges = large_test_data

    kwargs = {
        'fit_params': LARGE_FIT_PARAMS,
        'covariates': covariates,
        'cases': cases,
        'discharges': discharges,
        'dist_params': FULL_DIST_PARAMS
    }
    # Ensure that numba can jit the function before timing it
    likelihood.carehome_intensity(**kwargs)
    benchmark(likelihood.carehome_intensity, **kwargs)


def test_likelihood_performance(large_test_data, benchmark):
    '''
    Test the performance of the calculation of likelihood from the intensity
    and case distribution.'''

    _, cases, covariates, discharges = large_test_data
    intensity = likelihood.carehome_intensity(
        fit_params=LARGE_FIT_PARAMS,
        covariates=covariates,
        cases=cases,
        discharges=discharges,
        dist_params=FULL_DIST_PARAMS
    )

    benchmark(likelihood.likelihood, intensity, cases)
