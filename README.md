# Bayesian fitting of care home data in SAIL

## Structure

* `generate_sample_data.py`: tool to generate sample data of the form
  expected by the other tooling. Should approximately meet the specifications
  of the data, but distributions are uniform so don't expect to see the
  same observed behaviour as reality. Has decent test coverage.
* `likelihood.py`: implements the log-likelihood calculation using Numpy.
  Good test coverage.
* `mcmc.py`: maximimises the likelihood based on three cases: one with only the
  baseline intensity, one with the baseline plus case self-excitation, and
  finally one with the baseline plus self- and hospital discharge-induced
  excitations. Currently no unit tests in place.

## Installation

Packages required are listed in `requirements.txt`. These can be installed
into a new virtual environment using

    pip install -r requirements.txt

Most requirements are included with Anaconda; the remainder can also be
installed into a Conda environment using `conda` instead of `pip`.

## Usage

### Data

To generate some sample data to check the usage of the tools outside of the
SAIL environment:

    make sample_data
    
This creates a sampledata directory containing full-sized and medium-sized
data sets.

Each data set is expected to comprise three CSV files:

 * cases: one column per care home, a header row containing care home IDs, 
   followed by one row per day of integers representing number of new cases
   in that care home on that day.
 * covariates: one column per case home, a header row containing care home IDs,
   followed by one row containing integers representing a banded classifcation
   of care home size. This index should start at zero. Currently a
   classification 0-3 is tested, but other maximum indices may work, and if not
   then this would be easy to correct.
 * discharges: one column per care home, a header row containing care home IDs, 
   followed by one row per day of integers representing number of hospital
   discharges into that care home on that day.

### Likelihood

To calculate the likelihood of a particular parameter set, use the command

    python likelihood.py

with an appropriate set of arguments. The minimal set of parameters is:

* `--cases_file`, followed by the filename of the cases CSV file (for example,
  `--cases_file sampledata/cases_medium.csv`)
* `--covariates_file`, followed by the filename of the covariates CSV file
  (for example, `--covariates_file sampledata/covariates_medium.csv`)
* `--baseline_intensities`, followed by a list of the baseline intensities tool
  calculate with (for example, `--baseline_intensities 0.2 0.4 0.6 0.8`)

Optional parameters:

* `--discharges_file`, followed by the filename of the cases CSV file (for
  example, `--discharges_file sampledata/discharges_medium.csv`).
  Must be specified if `r_h` is non-zero.
* `--r_c` or `--r_h`, followed by the coefficient associated with the self- and
  discharge excitation respectively (for example, `--r_c 1.5 --r_h 0.5`).
  By default these terms are zero.
* `--self_excitation_mean` or `--discharge_excitation_mean`, followed by the
  mean of the gamma distribution of self- or discharge times respectively.
  Default is
  `--self_excitation_mean 6.5 --discharge_excitation_mean 6.5`
* `--self_excitation_cv` and `--discharge_excitation_cv`, followed by the
  coefficient of variation of the gamma distribution of self- or discharge
  times respectively.
  Default is `--self_excitation_cv 0.62 --discharge_excitation_cv 0.62`

### MCMC fit

To fit a given data set, use the command

    python mcmc.py

Many parameters behave the same as for `likelihood.py`. Of those that differ,
required parameters are:

* `--baseline_intensities` may be followed by a single integer representing
  the number of care home size classifications
* `--output_directory` specifies the directory in which to place the output.
  This should not already exist (unless the `--overwrite` option is specified.)

Optional parameters:

* `--overwrite`: if the output directory already existss, then remove it.
  *Use with caution.*
* `--num_burn`, `--num_draws`, followed by an integer will set the number of
  thermalisation samples and the number of samples drawn from the posterior
  distribution respectively.

This will display summary results on screen, and also create files in the
output directory. Filenames start with `base`, `self`, or `full`; `base` and `self` are created in all cases, while `full` is only created if discharge data
are supplied. For each case, three objects are created:

* `summary.txt`, containing the same statistics output to the screen, namely
  the mean, standard deviation, confidence intervals, etc. for the fit
  parameters estimated from the posterior distributions.
* `trace.dat`, a directory containing traces from the Monte Carlo simulation,
  which allows further analysis of the output if necessary.
* `traceplot.pdf`, containing plots of the traces and their histograms, to
  allow a judgement of the stability of the fit.

### Tests

The test suite can be run using

    make test

This performs unit tests verifying the expected behaviour of `likelihood.py`
and `generate_sample_data.py`. This should complete very quickly; skipped
tests are the performance benchmark tests mentioned below which take longer.

### Other utilities

The included `Makefile` provides some convenience tools:

* `make clean`: remove the generated sample data
* `make benchmark`: run timings of the key functions in `likelihood.py` using
  representatively large sample data. This makes use of `pytest-benchmark`;
  skipped tests are the unit tests, which are not benchmarked.
* `make benchmark_record`: run the above timings, and also save the results
  so that they can be compared later using `pytest-benchmark compare`.
* `make clean_timings`: remove the saved benchmark timings.`
