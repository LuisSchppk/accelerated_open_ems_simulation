import multiprocessing
import os
import timeit
from datetime import datetime
import fast_cycle_loop

import pandas as pd

from evaluation import evaluate_and_store_final
from config import main_config, support_config
from multiprocessing import Pool
from const import date_format, default_min_energy, consumption_time_resolution, production_time_resolution, df_columns, \
    high_grid_limit, low_grid_limit, f_result_path, summer_str, winter_str, high_grid_limit_str, low_grid_limit_str, \
    worker_count
from cycle import setup_simulation_java
from utils import id_to_site_and_remainder


def run_simulations(p_start_string, p_sim_id, production_array, consumption_array, grid_limit,
                    p_simulation_time, p_chunk_size):
    time_start = datetime.strptime(p_start_string, date_format)
    site, remainder = id_to_site_and_remainder(p_sim_id)
    result_path = f_result_path.format(site, remainder)
    try:
        now = str(datetime.now()).replace(':', '_')
        os.rename(f'{result_path}', f'{result_path}_{now}')
    except FileNotFoundError:
        pass

    cycle_worker = setup_simulation_java(start=time_start,
                                         main_config=main_config,
                                         support_config=support_config,
                                         grid_limit=grid_limit,
                                         min_energy=default_min_energy)

    fast_cycle_loop.fast_cycle_loop_java(simulation_time=p_simulation_time,
                                         cycle_worker=cycle_worker,
                                         production=production_array,
                                         consumption=consumption_array,
                                         consumption_time_resolution=consumption_time_resolution,
                                         production_time_resolution=production_time_resolution,
                                         columns=df_columns,
                                         chunk_size=p_chunk_size,
                                         sim_id=p_sim_id,
                                         )
    evaluate_and_store_final(sim_id=p_sim_id, grid_limit=grid_limit)


def preprocess_load_production(data, modifier):
    data = data.squeeze().to_numpy()
    data = data * modifier
    data = data.astype(int)
    return data


def all_parallel(sites, start_string, simulation_time, chunk_size):
    seasons = [summer_str, winter_str]
    grid_limits = [high_grid_limit_str, low_grid_limit_str]

    seasons = [summer_str]
    grid_limits = [high_grid_limit_str]

    summer_production = pd.read_csv(f'config/Production/Summer.csv')['Power']
    winter_production = pd.read_csv('config/Production/Winter.csv')['Power']

    summer_production = preprocess_load_production(data=summer_production, modifier=20)
    winter_production = preprocess_load_production(data=winter_production, modifier=1)

    items = list()
    sim_ids = list()
    for site in sites:
        consumption = pd.read_csv(f'config/Consumption/{site}.csv')['Power']
        consumption = preprocess_load_production(consumption, 1)
        simulation_time = min(len(consumption)*consumption_time_resolution, simulation_time)
        for season in seasons:
            if season is summer_str:
                production = summer_production
                simulation_time = min(simulation_time, len(summer_production)*production_time_resolution)
            elif season is winter_str:
                production = winter_production
                simulation_time = min(simulation_time, len(winter_production)*production_time_resolution)
            else:
                raise Exception(f'Unknown Season: {season}')
            for limit_type in grid_limits:
                if limit_type is high_grid_limit_str:
                    grid_limit = high_grid_limit
                elif limit_type is low_grid_limit_str:
                    grid_limit = low_grid_limit
                else:
                    raise Exception(f'Unknown Grid Limit Type: {limit_type}')
                sim_id = f'{site}_{season}_{limit_type}_Grid'
                sim_ids.append((sim_id, grid_limit))
                items.append([sim_id, start_string, production, consumption, grid_limit, simulation_time, chunk_size])

    total_start = timeit.default_timer()
    with Pool(worker_count) as p:
        p.starmap(process_single_simulation, items)  # per site: 60s. per sim: 36s
    total_stop = timeit.default_timer()
    print('Total Calculation Time: ', end=' ')
    print(total_stop - total_start)


def all_mixed(sites, start_string, simulation_time, chunk_size):
    summer_production = pd.read_csv(f'config/Production/Summer.csv')['Power']
    winter_production = pd.read_csv('config/Production/Winter.csv')['Power']
    summer_production = preprocess_load_production(data=summer_production, modifier=20)
    winter_production = preprocess_load_production(data=winter_production, modifier=1)
    seasons = [summer_str, winter_str]
    grid_limits = [high_grid_limit_str, low_grid_limit_str]
    total_start = timeit.default_timer()

    for site in sites:
        consumption = pd.read_csv(f'config/Consumption/{site}.csv')['Power']
        consumption = preprocess_load_production(consumption, 1)
        items = list()
        sim_ids = list()

        for season in seasons:
            if season is summer_str:
                production = summer_production
            elif season is winter_str:
                production = winter_production
            else:
                raise Exception(f'Unknown Season: {season}')
            for limit_type in grid_limits:
                if limit_type is high_grid_limit_str:
                    grid_limit = high_grid_limit
                elif limit_type is low_grid_limit_str:
                    grid_limit = low_grid_limit
                else:
                    raise Exception(f'Unknown Grid Limit Type: {limit_type}')
                sim_id = f'{site}_{season}_{limit_type}_Grid'
                sim_ids.append((sim_id, grid_limit))
                items.append([sim_id, start_string, production, consumption, grid_limit, simulation_time, chunk_size])

        with Pool(worker_count) as p:
            p.starmap(process_single_simulation, items)  # per site: 60s. per sim: 36s
    total_stop = timeit.default_timer()
    print('Total Calculation Time: ', end=' ')
    print(total_stop - total_start)


def all_sequential(sites, start_string, simulation_time, chunk_size):
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
        run_simulations(p_start_string=start_string,
                        p_sim_id=sim_id,
                        production_array=summer_production,
                        consumption_array=consumption,
                        grid_limit=high_grid_limit,
                        p_simulation_time=simulation_time, p_chunk_size=chunk_size)
        stop = timeit.default_timer()
        print(stop - start)

        # Summer Low Grid
        sim_id = f'{site}_Summer_Low_Grid'
        print(f'Run {sim_id}')
        start = timeit.default_timer()
        run_simulations(p_start_string=start_string,
                        p_sim_id=sim_id,
                        production_array=summer_production,
                        consumption_array=consumption,
                        grid_limit=low_grid_limit,
                        p_simulation_time=simulation_time, p_chunk_size=chunk_size)
        stop = timeit.default_timer()
        print(stop - start)

        # Summer High Grid
        sim_id = f'{site}_Winter_High_Grid'
        print(f'Run {sim_id}')
        start = timeit.default_timer()
        run_simulations(p_start_string=start_string,
                        p_sim_id=sim_id,
                        production_array=winter_production,
                        consumption_array=consumption,
                        grid_limit=high_grid_limit,
                        p_simulation_time=simulation_time, p_chunk_size=chunk_size)
        stop = timeit.default_timer()
        print(stop - start)

        # Summer High Grid
        sim_id = f'{site}_Winter_Low_Grid'
        print(f'Run {sim_id}')
        start = timeit.default_timer()
        run_simulations(p_start_string=start_string,
                        p_sim_id=sim_id,
                        production_array=winter_production,
                        consumption_array=consumption,
                        grid_limit=low_grid_limit,
                        p_simulation_time=simulation_time, p_chunk_size=chunk_size)
        stop = timeit.default_timer()
        print(stop - start)


def process_single_simulation(sim_id, start_string, production, consumption,
                              grid_limit, simulation_time, chunk_size):
    print(f'Run {sim_id}')
    start = timeit.default_timer()
    run_simulations(p_start_string=start_string,
                    p_sim_id=sim_id,
                    production_array=production,
                    consumption_array=consumption,
                    grid_limit=grid_limit,
                    p_simulation_time=simulation_time, p_chunk_size=chunk_size)
    stop = timeit.default_timer()
    print(stop - start)


def process_site(site_name, start_string, summer_production, winter_production, chunk_size, simulation_time):
    consumption = pd.read_csv(f'config/Consumption/{site_name}.csv')['Power']
    consumption = preprocess_load_production(consumption, 1)

    # Summer High Grid
    sim_id = f'{site_name}_Summer_High_Grid'
    print(f'Run {sim_id}')
    start = timeit.default_timer()
    run_simulations(p_start_string=start_string,
                    p_sim_id=sim_id,
                    production_array=summer_production,
                    consumption_array=consumption,
                    grid_limit=high_grid_limit,
                    p_simulation_time=simulation_time, p_chunk_size=chunk_size)
    stop = timeit.default_timer()
    print(stop - start)

    # Summer Low Grid
    sim_id = f'{site_name}_Summer_Low_Grid'
    print(f'Run {sim_id}')
    start = timeit.default_timer()
    run_simulations(p_start_string=start_string,
                    p_sim_id=sim_id,
                    production_array=summer_production,
                    consumption_array=consumption,
                    grid_limit=low_grid_limit,
                    p_simulation_time=simulation_time, p_chunk_size=chunk_size)
    stop = timeit.default_timer()
    print(stop - start)

    # Summer High Grid
    sim_id = f'{site_name}_Winter_High_Grid'
    print(f'Run {sim_id}')
    start = timeit.default_timer()
    run_simulations(p_start_string=start_string,
                    p_sim_id=sim_id,
                    production_array=winter_production,
                    consumption_array=consumption,
                    grid_limit=high_grid_limit,
                    p_simulation_time=simulation_time, p_chunk_size=chunk_size)
    stop = timeit.default_timer()
    print(stop - start)

    # Summer High Grid
    sim_id = f'{site_name}_Winter_Low_Grid'
    print(f'Run {sim_id}')
    start = timeit.default_timer()
    run_simulations(p_start_string=start_string,
                    p_sim_id=sim_id,
                    production_array=winter_production,
                    consumption_array=consumption,
                    grid_limit=low_grid_limit,
                    p_simulation_time=simulation_time, p_chunk_size=chunk_size)
    stop = timeit.default_timer()
    print(stop - start)
