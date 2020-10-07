#!/usr/bin/env python

import pytest
import generate_sample_data
import likelihood


@pytest.mark.parametrize(
    'parameter_set,nrows,ncols',
    [('full', 182, 1000), ('medium', 19, 100)]
)
def test_generate(parameter_set, nrows, ncols, tmp_path):
    '''
    Check that sample data files are generated.
    '''

    generate_sample_data.generate(parameter_set, output_dir=tmp_path)

    for prefix, length in [
            ('covariates', 2), ('cases', nrows), ('discharges', nrows)
    ]:
        line_index = -1
        for line_index, line in enumerate(open(
                tmp_path / f'{prefix}_{parameter_set}.csv'
        )):
            assert len(line.split(',')) == ncols
        assert line_index == length - 1


@pytest.mark.parametrize(
    'parameter_set,n_cases,n_case_homes,n_discharges,n_discharge_homes',
    [('full', 2000, 330, 3000, 500), ('medium', 200, 33, 300, 50)]
)
def test_loadable(parameter_set, n_cases, n_case_homes, n_discharges,
                  n_discharge_homes, tmp_path):
    '''
    Check that our functions in likelihood can read the sample data, and that
    the numbers add up to what they are supposed to.
    '''

    generate_sample_data.generate(parameter_set, output_dir=tmp_path)

    ch_ids, cases, covariates, discharges = (
        likelihood.safely_read_cases_covariates_discharges(
            tmp_path / f'cases_{parameter_set}.csv',
            tmp_path / f'covariates_{parameter_set}.csv',
            tmp_path / f'discharges_{parameter_set}.csv'
        )
    )

    assert cases.sum() == n_cases
    assert discharges.sum() == n_discharges
    assert (ch_ids < generate_sample_data.MAX_CAREHOME_ID).all()
    assert (covariates < generate_sample_data.MAX_CATEGORIES).all()

    # Since cases are randomly assigned to a subset of homes, it's possible
    # for a couple to end up with no cases after all. This isn't a
    # huge concern currently, but could be tightened up if necessary.
    assert (
        (cases.sum(axis=0) > 0).sum() > n_case_homes - n_case_homes ** 0.5 / 3
    )
    assert (
        (discharges.sum(axis=0) > 0).sum()
        > n_discharge_homes - n_discharge_homes ** 0.5 / 3
    )
