#!/usr/bin/env python

'''
A generator for some arbitrary test data roughly matching the description
of the expected data from SAIL.

Notes:
* Additional covariate: Baseline intensity varies by care home size.
* Data is daily number of cases from February to July.
* 1000 Care homes
* 330 homes had cases, 670 no cases
* 2000 total cases
* 50% homes received discharge transfer from hospital
* 3000 discharges in total
* Data Frame 1: 1 row per day, 1 column per care home ID, entries are numbers
  of cases
* Data Frame 2: 1 row per day, 1 column per care home ID, entries are numbers
  of discharges
* Data frame 3: 1 row per covariate, 1 column per care home ID, entries are
  covariates (to start with just one, care home size, likely as a category with
  4 groups).
'''

from pathlib import Path
import numpy as np


MAX_BEDS_PER_HOME = 255
MAX_CATEGORIES = 4

MAX_CAREHOME_ID = 32767

PARAMETERS = {
    'full': {
        'n_care_homes': 1000,
        'n_cases': 2000,
        'n_case_homes': 330,
        'n_discharges': 3000,
        'n_discharge_homes': 500,
        'n_days': 181,
        'n_covariates': 1
    },
    'medium': {
        'n_care_homes': 100,
        'n_cases': 200,
        'n_case_homes': 33,
        'n_discharges': 300,
        'n_discharge_homes': 50,
        'n_days': 18,
        'n_covariates': 1
    }
}


def generate(parameter_set, output_dir=Path('.')):
    '''
    Generate sample data for a given parameter_set and save it to disk.
    '''

    p = PARAMETERS[parameter_set]
    cases = np.zeros((p['n_days']+1, p['n_care_homes']), dtype=np.int32)
    discharges = np.zeros((p['n_days']+1, p['n_care_homes']), dtype=np.int32)
    covariates = np.zeros((p['n_covariates']+1, p['n_care_homes']), dtype=np.int32)

    rng = np.random.default_rng()
    care_home_ids = rng.choice(MAX_CAREHOME_ID, size=p['n_care_homes'], replace=False)

    for sample_array in cases, discharges, covariates:
        sample_array[0] = rng.permutation(care_home_ids)

    for sample_array, num_instances, num_places in (
            (cases, p['n_cases'], p['n_case_homes']),
            (discharges, p['n_discharges'], p['n_discharge_homes'])
    ):
        for _ in range(num_instances):
            sample_array[1 + rng.integers(p['n_days']),
                         rng.integers(num_places)] += 1

    covariates[1] = rng.choice(MAX_CATEGORIES, size=p['n_care_homes'])

    np.savetxt(output_dir / f"covariates_{parameter_set}.csv", covariates,
               fmt="%d", delimiter=',')
    np.savetxt(output_dir / f"cases_{parameter_set}.csv", cases,
               fmt="%d", delimiter=',')
    np.savetxt(output_dir / f"discharges_{parameter_set}.csv", discharges,
               fmt="%d", delimiter=',')


def main():
    '''Parse the command line for which parameter set to run and then run it'''

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('parameter_set')
    parser.add_argument('--output_dir', default=Path('.'), type=Path)
    args = parser.parse_args()

    if args.parameter_set not in PARAMETERS:
        raise ValueError('Parameter set is not valid')

    generate(args.parameter_set, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
