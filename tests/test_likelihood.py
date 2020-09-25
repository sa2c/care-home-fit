#!/usr/bin/env python

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import likelihood


SMALL_FIT_PARAMS = {
    'baseline_intensities': np.asarray([1, 2, np.nan, np.nan]),
    'r_h': 1.5,
    'r_c': 0.5
}

SMALL_DIST_PARAMS = {
    'self_excitation_shape': 2.6,
    'self_excitation_scale': 2.5,
    'discharge_excitation_shape': 2.6,
    'discharge_excitation_scale': 2.5
}


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
    ids, values = likelihood.read_and_tidy_data('tests/fixtures/small.csv')
    assert_array_equal(ids, [14, 16, 35])
    assert_array_equal(
        values,
        [[4, 1, 6], [4, 0, 3], [6, 66, 2]]
    )

@pytest.fixture
def small_cases():
    return likelihood.read_and_tidy_data('tests/fixtures/small.csv')


@pytest.fixture
def small_covariates():
    return likelihood.read_and_tidy_data('tests/fixtures/small_covariates.csv')


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


@pytest.mark.parametrize("mean, cv, expected_shape, expected_scale",
                         [(1, 1, 1, 1), (6.5, 0.62, 2.601, 2.499)])
def test_calculate_gamma_parameters(mean, cv, expected_shape, expected_scale):
    '''Test that calculation of Scipy-style gamma parameters from "descriptive"
    gamma parameters is correct.'''
    shape, scape = likelihood.calculate_gamma_parameters(mean, cv)
    assert_almost_equal([shape, scape], [expected_shape, expected_scale],
                        decimal=3)
