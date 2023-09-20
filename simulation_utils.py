import multiprocessing
import os
import timeit
from datetime import datetime

import numpy as np

import fast_cycle_loop

import pandas as pd

from evaluation import evaluate_and_store_final
from config import main_config, support_config, production_modifiers, consumption_modifiers
from multiprocessing import Pool
from const import date_format, df_columns
from config import default_min_energy, consumption_time_resolution, production_time_resolution, \
    high_grid_limit, low_grid_limit, f_result_path, summer_str, winter_str, high_grid_limit_str, low_grid_limit_str, \
    worker_count, grid_limits
from cycle import setup_simulation_java
from utils import id_to_site_and_season_grid


def run_simulations(p_start_string, p_sim_id, production_array, consumption_array, grid_limit,
                    p_simulation_time, p_chunk_size):
    """
    Execute single simulation with id <p_sim_id>.

    :param p_start_string: Starting date for simulation given in format yyyy-mm-ddTHH:MM:SSZ.
    :param p_sim_id: Simulation ID to name result files according to simulation and scenario.
    :param production_array: np.ndarray containing the production values in [W] that will be used for the simulation.
    :param consumption_array: np.ndarray containing the consumption values in [W] that will be used for the simulation.
    :param grid_limit: Amount of [W] drawn from the grid, that should not be exceeded.
    :param p_simulation_time: Duration of simulation given in [s].
    :param p_chunk_size: Chunk size in [s] for memory optimization.
                        The total simulation time is divided in chunks of size <chunk_size>.
                        After on chunk has been processed, the intermediary results will be stored on the hard drive.
    :return:
    """

    time_start = datetime.strptime(p_start_string, date_format)

    # Construct file name for storing results.
    site, season_grid_str = id_to_site_and_season_grid(p_sim_id)
    result_path = f_result_path.format(site, season_grid_str)
    try:

        # If results for the simulation already exists. Rename old results by appending date, to free up name.
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
                                         column_count=len(df_columns),
                                         chunk_size=p_chunk_size,
                                         sim_id=p_sim_id,
                                         )

    # Evaluate and store the final results, made up by the intermediary results.
    evaluate_and_store_final(sim_id=p_sim_id, grid_limit=grid_limit)


def preprocess_load_production(data: pd.Series, modifier: int) -> np.ndarray:
    """
    Convert input pd.Series data to int np.ndarray and multiply values by modifier.
    :param data: Series containing the load data in [W]
    :param modifier: Each entry in modifier gets multiplied by this value.
    :return:
    """
    data = data.squeeze().to_numpy()
    data = data * modifier
    data = data.astype(int)
    return data


def all_parallel(sites, seasons, grid_limits_str, start_string, simulation_time, chunk_size):
    """
    Execute all the simulations in parallel. The simulations are made up by all the combinations of the elements of
    sites, seasons and grid_limits_str.

    :param sites: String identifiers for the sites (i.e., consumption input), that will be used.
                  For an entry a respective csv file with name <entry>.csv in config/Consumption must exist.
    :param seasons: String identifiers for the seasons (i.e., production input), that will be used.
                  For an entry a respective csv file with name <entry>.csv in config/Consumption must exist.
    :param grid_limits_str: String identifiers for the grid limits, that will be used. Any entry has to be the key for
                            the dict grid_limits in config.py. The respective values can be adjusted there.
    :param start_string: Starting date for simulation given in format yyyy-mm-ddTHH:MM:SSZ
    :param simulation_time: Duration of simulation given in [s]. If any input (consumption or production)
                            spans a duration shorter than simulation_time, the simulated time will be set
                            to that duration for the affected simulations.
    :param chunk_size:  Chunk size in [s] for memory optimization.
                        The total simulation time is divided in chunks of size <chunk_size>.
                        After on chunk has been processed, the intermediary results will be stored on the hard drive.
    :return:
    """

    items = list()

    # Simulation IDs to name result files according to simulation and scenario
    sim_ids = list()
    for site in sites:
        consumption = pd.read_csv(f'config/Consumption/{site}.csv')['Power']
        consumption_modifier = consumption_modifiers.get(site)
        if consumption_modifier is None:
            consumption_modifier = 1
        consumption = preprocess_load_production(consumption, 1)

        # Adjust simulation time to be at most as long as the shortest input.
        simulation_time = min(len(consumption) * consumption_time_resolution, simulation_time)
        for season in seasons:
            production = pd.read_csv(f'config/Production/{season}.csv')['Power']
            production_modifier = production_modifiers.get(season)
            if production_modifier is None:
                production_modifier = 1

            production = preprocess_load_production(data=production, modifier=production_modifier)

            # Adjust simulation time to be at most as long as the shortest input.
            simulation_time = min(simulation_time, len(production) * production_time_resolution)

            for limit_str in grid_limits_str:
                grid_limit = grid_limits.get(limit_str)
                if grid_limit is None:
                    raise Exception(f'Unknown Grid Limit Type: {limit_str}')

                # construct sim_id
                sim_id = f'{site}_{season}_{limit_str}_Grid'
                sim_ids.append((sim_id, grid_limit))

                # Build list of items as parameters for the task, performed by the parallel pool.
                items.append([sim_id, start_string, production, consumption, grid_limit, simulation_time, chunk_size])

    total_start = timeit.default_timer()

    # Start parallel execution of the function process_single_simulation with one entry line of items as parameter.
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
        simulation_time = min(len(consumption) * consumption_time_resolution, simulation_time)
        summer_simulation_time = min(simulation_time, len(summer_production) * production_time_resolution)
        winter_simulation_time = min(simulation_time, len(winter_production) * production_time_resolution)

        # Summer High Grid
        sim_id = f'{site}_Summer_High_Grid'
        print(f'Run {sim_id}')
        start = timeit.default_timer()
        run_simulations(p_start_string=start_string,
                        p_sim_id=sim_id,
                        production_array=summer_production,
                        consumption_array=consumption,
                        grid_limit=high_grid_limit,
                        p_simulation_time=summer_simulation_time, p_chunk_size=chunk_size)
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
                        p_simulation_time=summer_simulation_time, p_chunk_size=chunk_size)
        stop = timeit.default_timer()
        print(stop - start)

        # Winter High Grid
        sim_id = f'{site}_Winter_High_Grid'
        print(f'Run {sim_id}')
        start = timeit.default_timer()
        run_simulations(p_start_string=start_string,
                        p_sim_id=sim_id,
                        production_array=winter_production,
                        consumption_array=consumption,
                        grid_limit=high_grid_limit,
                        p_simulation_time=winter_simulation_time, p_chunk_size=chunk_size)
        stop = timeit.default_timer()
        print(stop - start)

        # Winter Grid
        sim_id = f'{site}_Winter_Low_Grid'
        print(f'Run {sim_id}')
        start = timeit.default_timer()
        run_simulations(p_start_string=start_string,
                        p_sim_id=sim_id,
                        production_array=winter_production,
                        consumption_array=consumption,
                        grid_limit=low_grid_limit,
                        p_simulation_time=winter_simulation_time, p_chunk_size=chunk_size)
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
