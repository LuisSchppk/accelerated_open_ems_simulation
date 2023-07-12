from const import summer_str, winter_str, high_grid_limit_str, low_grid_limit_str
from simulation_utils import all_parallel, all_sequential

if __name__ == '__main__':
    start_string = "2022-04-01T11:00:00Z"
    simulation_time = 29 * 24 * 3600 + 13 * 3600  # 29d 13h 16:15-18:45
    chunk_size = 24 * 3600
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

    sites = ['CS', 'H', 'NH', 'C', 'OC', 'TZE']
    # sites = ['NH']
    seasons = [summer_str, winter_str]
    grid_limits = [high_grid_limit_str, low_grid_limit_str]

    all_parallel(sites=sites, start_string=start_string, simulation_time=simulation_time, chunk_size=chunk_size)

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
