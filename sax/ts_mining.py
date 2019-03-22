import pandas as pd
from sax.sax_main import sax_slot_size

def read_energy_data():
    df = pd.read_csv('../data/energydata.csv', index_col='date')

    # we need to find failures, and to get the timestamp. In our case, this corresponds to cases where appliances are too high:
    # let's say that we label when applicances are > 800

    timefails = df[['Appliances']][df.Appliances > 800]
    timeseries = df.loc[:, 'lights':]

    return timeseries, timefails


def ts_mining(timeseries, timefails):
    '''
    Mine caracteristics patterns of interval values before the fail appears
    For now, we consider taking 100 points before the fail.
    We consider timeseries to be sampled equally, at the same time.
    :param timeseries: list of timeseries needed (panda dataframe)
    :param timefails: list of timestamps associated with fail
    :return:
    '''

    # we iterate through timeseries to have their sax representation

    # we mine them with seed_explore

    # we convert obtained pattern to original form to give to experts

    pass


read_energy_data()
