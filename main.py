import evaluation
from config import summer_str, winter_str, high_grid_limit_str, low_grid_limit_str
from simulation_utils import all_parallel, all_sequential, all_mixed

if __name__ == '__main__':
    # starting date for simulation given in format yyyy-mm-ddTHH:MM:SSZ
    start_string = "2022-04-01T11:00:00Z"

    # duration of simulation given in [s]
    simulation_time = 72 * 3600  # 29 * 24 * 3600 + 13 * 3600  # 29d 13h

    # Chunk size in [s] for memory optimization. The total simulation time is divided in chunks of size <chunk_size>.
    # After on chunk has been processed, the intermediary results will be stored on the hard drive.
    chunk_size = 72 * 3600

    # <sitename>.csv in folder config/Consumption will be read as the input for the consumption of the simulation.
    # They need the field headers 'Datetime' and 'Power'.
    # The time delta between entries can be defined in 'const.py' with variable consumption_time_resolution.
    sites = ['CS', 'H', 'NH', 'C', 'OC', 'TZE']

    # <seasons>.csv in folder config/Production will be read as the input for the consumption of the simulation.
    # They need the field headers 'Datetime' and 'Power'.
    # The time delta between entries can be defined in 'const.py' with variable production_time_resolution.
    seasons = [summer_str, winter_str]

    grid_limits = [high_grid_limit_str, low_grid_limit_str]

    # Parallel execution of all simulation. Each tuple of [site, season, grid_limit]
    # gets executed as a single parallel task.
    all_parallel(sites=sites, seasons=seasons, grid_limits_str=grid_limits, start_string=start_string,
                 simulation_time=simulation_time, chunk_size=chunk_size)

    # Note: all_parallel was the fastest for me and I mostly used it. Therefore, the two other execution methods have
    # not been refactored or commented in detail. They should work, but maybe need some updating.

    # Sequential execution of all simulations
    # all_sequential(sites=sites, start_string=start_string, simulation_time=simulation_time, chunk_size=chunk_size)

    # Mixed execution of all simulations. All simulations in one site get executed in parallel. The respective sites are
    # executed sequential.
    # all_mixed(sites=sites, start_string=start_string, simulation_time=simulation_time, chunk_size=chunk_size)
