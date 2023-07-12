import os
import re
import timeit

import pandas as pd

from const import date_time_str


def interval_to_duration(interval):
    unit = re.sub('[0-9]', '', interval)
    number = int(re.sub('[A-Za-z]', '', interval))
    if unit == 's':
        return number / 3600
    elif unit == 'm':
        return number / 60
    elif unit == 'h':
        return number
    else:
        raise Exception(f'Interval unit {unit} not supported. Try "s","m" or "h" instead.')


def append_to_csv(df, filename, index=True):
    with open(filename, 'a', newline='\n') as f:
        df.to_csv(f, mode='a', header=f.tell() == 0, index=index)


def append_to_fastparquet(df, filename, index=True):
    if not os.path.isfile(filename):
        df.to_parquet(filename, engine='fastparquet')
    else:
        df.to_parquet(filename, engine='fastparquet', append=True, index=index)


def append_series(list):
    result = list[0]
    if not isinstance(result, pd.Series):
        result = pd.Series(result)
    list = list[1:]
    for s in list:
        if not isinstance(s, pd.Series):
            s = pd.Series(s)
        result = pd.concat([result, s])
    return result


def read_cycles(cycles_path, sim_id):
    main_charge_cycles = pd.read_csv(f'{cycles_path}/main_charge_{sim_id}.csv')
    main_discharge_cycles = pd.read_csv(f'{cycles_path}/main_discharge_{sim_id}.csv')
    support_charge_cycles = pd.read_csv((f'{cycles_path}/support_charge_{sim_id}.csv'))
    support_discharge_cycles = pd.read_csv(f'{cycles_path}/support_discharge_{sim_id}.csv')
    cycle_counts = pd.read_csv(f'{cycles_path}/cycle_counts_{sim_id}.csv', index_col=date_time_str)
    cycle_counts.index = pd.to_datetime(cycle_counts.index)
    return main_charge_cycles, main_discharge_cycles, support_charge_cycles, support_discharge_cycles, cycle_counts


def id_to_site_and_remainder(sim_id):
    split = sim_id.split('_')
    site = split[0]
    remainder = '_'.join(split[1:])
    return site, remainder
