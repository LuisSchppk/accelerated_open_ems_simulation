import timeit
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import cython
import jnius_config

import evaluation
import fast_cycle_loop
from const import *
from java_classes import *
from config import *
from cycle import date_format, setup_simulation, setup_simulation_java


def run_simulations(start_string, sim_id, production, consumption, grid_limit, simulation_time, chunk_size):
    time_start = datetime.strptime(start_string, date_format)
    try:
        now = str(datetime.now()).replace(':', '_')
        os.rename(f'Results/{sim_id}', f'Results/{sim_id}_{now}')
    except FileNotFoundError:
        pass

    cycle_worker = setup_simulation_java(start=time_start,
                                         main_config=main_config,
                                         support_config=support_config,
                                         grid_limit=grid_limit,
                                         min_energy=default_min_energy)

    fast_cycle_loop.fast_cycle_loop_java(simulation_time=simulation_time,
                                         cycle_worker=cycle_worker,
                                         production=production,
                                         consumption=consumption,
                                         consumption_time_resolution=consumption_time_resolution,
                                         production_time_resolution=production_time_resolution,
                                         columns=df_columns,
                                         chunk_size=chunk_size,
                                         sim_id=sim_id,
                                         )
    evaluation.evaluate_and_store_final(sim_id=sim_id, grid_limit=grid_limit)


def preprocess_load_production(data, modifier):
    data = data.squeeze().to_numpy()
    data = data * modifier
    data = data.astype(int)
    return data


if __name__ == '__main__':
    start_string = "2022-04-01T11:00:00Z"
    simulation_time = 24 * 3600  # 29 * 24 * 3600 + 13 * 3600  # 29d 13h
    chunk_size = 12 * 3600

    # sites = ['CS', 'H', 'NH', 'C', 'OC', 'TZE']
    sites = ['OC', 'TZE']
    summer_production = pd.read_csv(f'config/Production/Summer.csv')['Power']
    winter_production = pd.read_csv('config/Production/Winter.csv')['Power']

    summer_production = preprocess_load_production(data=summer_production, modifier=20)
    winter_production = preprocess_load_production(data=winter_production, modifier=1)

    for site in sites:
        consumption = pd.read_csv(f'config/Consumption/{site}.csv')['Power']
        consumption = preprocess_load_production(consumption, 1)

        # Summer High Grid
        sim_id = f'{site}_Summer_High_Grid'
        print(f'Run {sim_id}')
        start = timeit.default_timer()
        run_simulations(start_string=start_string,
                        sim_id=sim_id,
                        production=summer_production,
                        consumption=consumption,
                        grid_limit=high_grid_limit,
                        simulation_time=simulation_time, chunk_size=chunk_size)
        stop = timeit.default_timer()
        print(stop - start)

        # Summer Low Grid
        sim_id = f'{site}_Summer_Low_Grid'
        print(f'Run {sim_id}')
        start = timeit.default_timer()
        run_simulations(start_string=start_string,
                        sim_id=sim_id,
                        production=summer_production,
                        consumption=consumption,
                        grid_limit=low_grid_limit,
                        simulation_time=simulation_time, chunk_size=chunk_size)
        stop = timeit.default_timer()
        print(stop - start)

        # Summer High Grid
        sim_id = f'{site}_Winter_High_Grid'
        print(f'Run {sim_id}')
        start = timeit.default_timer()
        run_simulations(start_string=start_string,
                        sim_id=sim_id,
                        production=winter_production,
                        consumption=consumption,
                        grid_limit=high_grid_limit,
                        simulation_time=simulation_time, chunk_size=chunk_size)
        stop = timeit.default_timer()
        print(stop - start)

        # Summer High Grid
        sim_id = f'{site}_Winter_Low_Grid'
        print(f'Run {sim_id}')
        start = timeit.default_timer()
        run_simulations(start_string=start_string,
                        sim_id=sim_id,
                        production=winter_production,
                        consumption=consumption,
                        grid_limit=low_grid_limit,
                        simulation_time=simulation_time, chunk_size=chunk_size)
        stop = timeit.default_timer()
        print(stop - start)

    #
    # consumption_str = 'NH'
    # production_str = 'Summer_09_07'

    # consumption = pd.read_csv(f'config/Consumption/{consumption_str}.csv', index_col=date_time_str)
    # production = pd.read_csv(f'config/Production/{production_str}.csv')
    #
    # grid_limit = high_grid_limit
    #
    # consumption, production = preprocessing(consumption, production)

    # 24h = 54s
    # result = fast_cycle_loop.fast_cycle_loop(simulation_time=simulation_time,
    #                                          cycle_worker=cycle_worker,
    #                                          production=production,
    #                                          consumption=consumption,
    #                                          consumption_time_resolution=consumption_time_resolution,
    #                                          production_time_resolution=production_time_resolution,
    #                                          columns=columns
    #                                          )

    # 24= 18s
    # result = fast_cycle_loop.fast_cycle_loop_java(simulation_time=simulation_time,
    #                                               cycle_worker=cycle_worker,
    #                                               production=production,
    #                                               consumption=consumption,
    #                                               consumption_time_resolution=consumption_time_resolution,
    #                                               production_time_resolution=production_time_resolution,
    #                                               columns=df_columns,
    #                                               chunk_size=simulation_time
    #                                               )
    # 4h = 24s
    # for cycle_count in range(simulation_time):
    #     if cycle_count % production_time_resolution == 0:
    #         idx = int(cycle_count / production_time_resolution)
    #         current_production = production[idx]
    #     if cycle_count % consumption_time_resolution == 0:
    #         idx = int(cycle_count / consumption_time_resolution)
    #         current_consumption = consumption[idx]
    #     tmp = cycle_worker.execute_cycle(current_production, current_consumption)
    #     result.loc[cycle_count] = tmp
