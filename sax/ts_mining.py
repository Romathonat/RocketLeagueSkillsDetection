import pandas as pd
from sax.sax_main import sax_slot_size

def read_energy_data():
    df = pd.read_csv('../data/energydata.csv', index_col='date')

    # we need to find failures, and to get the timestamp. In our case, this corresponds to cases where appliances are too high:
    # let's say that we label when applicances are > 800

    timefails = df[['Appliances']][df.Appliances > 700]
    timeseries = df.loc[:, 'lights':]

    return timeseries, timefails

def sax_to_fail_sequences(sax_timeseries, timefails):
    '''
    :param sax_timeseries:
    :param timefails: pandas serie with fails in the order of time
    :return:
    '''
    for index, row in sax_timeseries:
        pass


def ts_mining(timeseries, timefails, nb_points_slot, a, nb_point_back):
    '''
    Mine caracteristics patterns of interval values before the fail appears.
    For now, we consider taking 100 points before the fail.
    We consider timeseries to be sampled equally, at the same time.
    :param timeseries: list of timeseries needed (panda dataframe)
    :param timefails: list of timestamps associated with fail
    :return:
    '''

    # we iterate through timeseries to have their sax representation
    sax_timeseries = []

    for ts in timeseries:
        sax_rpz = sax_slot_size(timeseries[ts], nb_points_slot, a)
        sax_timeseries.append(sax_rpz)

    print(sax_timeseries)

    # we capture the moment where we have a fail: we go back in time, tacking nb_point_back before the fail



    # we mine them with seed_explore


    # we convert obtained pattern to original form to give to experts

    pass

timeseries, timefails = read_energy_data()

print(ts_mining(timeseries, timefails, 5, 10, 10))
