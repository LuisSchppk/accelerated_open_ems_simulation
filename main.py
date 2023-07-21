import evaluation
from const import summer_str, winter_str, high_grid_limit_str, low_grid_limit_str, high_grid_limit
from simulation_utils import all_parallel, all_sequential

if __name__ == '__main__':
    start_string = "2022-04-01T11:00:00Z"
    simulation_time = 29 * 24 * 3600 + 13 * 3600  # 29d 13h
    chunk_size = 72 * 3600

    sites = ['CS', 'H', 'NH', 'C', 'OC', 'TZE']

    # sites = ['CS']

    # evaluation.evaluate_and_store_final(sim_id='CS_Winter_Low_Grid',grid_limit=high_grid_limit)
    all_parallel(sites=sites, start_string=start_string, simulation_time=simulation_time, chunk_size=chunk_size)

    # pypy
    # full time:
    # 96->255s.
    # 96-> 271s
    # 48:  12-> 19s; 24->19s; 48-> 17s;
    # 96: 48-> 36s; 96->34s, 24->37s
    # 120: 60-> 44s; 30-> 47s; 120-> 44s
    # parallel:
    # 48: 12->72s; 24->72s
    # parallel 4core:
    # 48: 24 -> 158
    # parallel 8core
    # 48: 24-> 95s
    # parallel 12core
    # 48: 24-> 88s
    # mixed:
    # 46: 24-> 159