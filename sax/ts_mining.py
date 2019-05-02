import random
import pandas as pd
from sax.sax_main import sax_slot_size

def read_energy_data():
    df = pd.read_csv('../data/energydata.csv', index_col='date')

    # we need to find failures, and to get the timestamp. In our case, this corresponds to cases where appliances are too high:
    # let's say that we label when applicances are > 800

    timefails = df[['Appliances']][df.Appliances > 700]
    timeseries = df.loc[:, 'lights':]

    return timeseries, timefails


def sax_to_fail_sequences(df_sax, timefails, nb_events_back):
    '''
    Find fails, take events before. This function also extract as many sequences with no fail, to balance data.

    :param df_sax:
    :param timefails: pandas serie with fails in the order of time
    :return: sequences to be ingested by seed_explore
    '''

    current_i_fail = 0
    current_fail = timefails.index.values[current_i_fail]

    fail_sequences = []
    normal_sequences = []

    count_without_fail = 0

    for row in df_sax.itertuples():
        if row.Index > current_fail:
            # we just passed the fail
            # we take points before
            fail_sequences.append(df_sax.iloc[-nb_events_back:])

            current_i_fail += 1

            # we stop if we found all fails
            if current_i_fail >= len(timefails):
                break

            current_fail = timefails.index.values[current_i_fail]
        else:
            count_without_fail += 1
            if count_without_fail == nb_events_back:
                normal_sequences.append(df_sax.iloc[-nb_events_back:])
                count_without_fail = 0


    # we add random normal sequences
    all_sequences = fail_sequences

    # should do it with a proper python exception but no time
    if len(fail_sequences) > normal_sequences:
        print("ERROR: Number of normal sequences is too LOW !")

    all_sequences.extend(random.sample(normal_sequences, len(fail_sequences)))

    # now we can remove the time, and create itemsets

    return all_sequences



def ts_mining(timeseries, timefails, nb_points_slot, a, nb_events_back):
    '''
    Mine caracteristics patterns of interval values before the fail appears.
    For now, we consider taking 100 points before the fail.
    We consider timeseries to be sampled equally, at the same time.
    :param timeseries: list of timeseries needed (panda dataframe)
    :param timefails: list of timestamps associated with fail
    :return:
    '''

    # we iterate through timeseries to have their sax representation
    df_sax = None

    for ts in timeseries:
        sax_rpz = sax_slot_size(timeseries[ts], nb_points_slot, a)

        if df_sax is None:
            df_sax = sax_rpz
        else:
            df_sax = pd.concat([df_sax, sax_rpz], axis=1)

    #print(df_sax)

    # we capture the moment where we have a fail: we go back in time, tacking nb_point_back before the fail
    sequences = sax_to_fail_sequences(df_sax, timefails, nb_events_back)

    print(len(sequences))
    print(len(timefails))

    # we mine them with seed_explore

    # we convert obtained pattern to original form to give to experts

    pass

# TODO: check ecg https://www.kaggle.com/c/seizure-prediction/data
# https://www.kaggle.com/c/belkin-energy-disaggregation-competition/data

timeseries, timefails = read_energy_data()

print(ts_mining(timeseries, timefails, 5, 10, 10))
