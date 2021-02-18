from numpy import arange, asarray, nan, zeros
import matplotlib.pyplot as plt
import likelihood


def get_events_from_definitions(definitions):
    events = zeros((50, len(definitions)))
    for column, (_, events_to_add) in enumerate(definitions.items()):
        for day, count in events_to_add:
            events[day, column] += count
    return events


def plot(
        covariates, cases, fit_params, filename, definitions, discharges=None
):
    '''Plot the given cases and the '''
    dist_params = {
        'self_excitation_shape': 2.6,
        'self_excitation_scale': 2.5,
        'discharge_excitation_shape': 2.6,
        'discharge_excitation_scale': 2.5
    }

    intensity = likelihood.carehome_intensity(
        cases=cases,
        fit_params=fit_params,
        covariates=covariates,
        dist_params=dist_params,
        discharges=discharges
    )

    fig, ax = plt.subplots()
    ax.set_xlabel('Days')
    ax.set_ylabel('Intensity')
    ax_right = ax.twinx()
    ax_right.set_ylabel('Number of events')

    markers = 'ox+*^'
    linestyles = ['-', '--', ':', '-.', (0, (4, 1, 1, 1, 1, 1))]

    for column, (marker, linestyle, (label, original_cases)) in enumerate(zip(
            markers, linestyles, definitions.items()
    )):
        colour = f'C{column}'
        ax.plot(intensity[:, column], color=colour, ls=linestyle)
        if original_cases:
            ax_right.scatter(
                *zip(*original_cases), color=colour, marker=marker, ls='None'
            )
        else:
            marker = 'None'
        ax.plot(nan, color=colour, marker=marker, label=label, ls=linestyle)

    ax.set_ylim((0, None))
    ax_right.set_ylim((0, None))
    max_num_events = int(ax_right.get_ylim()[1])
    num_num_event_ticks = 1 + min(5, max_num_events)
    num_event_tick_spacing = max_num_events // (num_num_event_ticks - 1)
    num_event_ticks = arange(0, max_num_events + 1, num_event_tick_spacing)
    ax_right.set_yticks(num_event_ticks)

    if len(definitions) > 4:
        legend_columns = 2
    else:
        legend_columns = 1

    ax.legend(loc='best', frameon=False, ncol=legend_columns)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def example1():
    fit_params = {
        'baseline_intensities': asarray([0.01]),
        'r_c': 0.6
    }
    covariates = asarray([[0, 0, 0]])

    case_definitions = {
        'Single case': [(3, 1)],
        'Triple case': [(4, 3)],
        'Cluster of three cases': [(5, 1), (6, 1), (8, 1)]
    }
    plot(covariates, get_events_from_definitions(case_definitions), fit_params,
         'example_intensities.pdf', case_definitions)


def example2():
    fit_params = {
        'baseline_intensities': asarray([0.0002, 0.00135, 0.004, 0.00694]),
        'r_c': 0.6,
        'r_h': 0.011
    }
    covariates = asarray([[0, 1, 2, 2, 3]])
    discharge_definitions = {
        'Q1, weekly discharges': [
            (day, 1) for day in range(1, 50, 7)
        ],
        'Q2, weekly discharges': [
            (day, 1) for day in range(1, 50, 7)
        ],
        'Q3, weekly discharges': [
            (day, 1) for day in range(1, 50, 7)
        ],
        'Baseline Q3': [],
        'Baseline Q4': [],
    }
    plot(covariates, zeros((50, 5)), fit_params, 'excited_intensities_simple.pdf',
         discharge_definitions,
         discharges=get_events_from_definitions(discharge_definitions))


def main():
    example1()
    example2()

if __name__ == '__main__':
    main()
