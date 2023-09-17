import multiprocessing

from java_classes import Config

main_capacity = 400_000
support_capacity = 220_000

main_config = Config('ess0',  # id
                     'main',  # name
                     True,  # OnGrid
                     main_capacity,  # Capacity
                     1_200,  # Ramp Rate [W/s]
                     10_000,  # Response Time [ms]
                     10_000,  # Inactivity Time [ms]
                     10,  # Min. SoC
                     90,  # Max. SoC
                     50,  # Initial SoC
                     90_000,  # Discharge Power
                     78_000,  # Charge Power
                     [17, 70],  # lowerSocBorder
                     [20, 70]  # upperSocBorder
                     )

support_config = Config('ess1',  # id
                        'support',  # name
                        True,  # OnGrid
                        support_capacity,  # Capacity
                        100_000,  # Ramp Rate
                        2_000,  # Response Time
                        10_000,  # Inactivity Time
                        5,  # Min. SoC
                        95,  # Max. SoC
                        50,  # Initial SoC
                        276_000,  # Discharge Power
                        276_000,  # Charge Power
                        [17, 50],  # lowerSocBorder
                        [20, 50]  # upperSocBorder
                        )

summer_str = 'Summer'
winter_str = 'Winter'

high_grid_limit_str = 'High'
low_grid_limit_str = 'Low'

low_grid_limit = 100_000
high_grid_limit = 200_000

grid_limits = {high_grid_limit_str: high_grid_limit,
               low_grid_limit_str: low_grid_limit}

default_min_energy = 50_000

# Time delta in [s] between to entries of the input consumption data.
consumption_time_resolution = 1 * 60  # 1min

# Time delta in [s] between to entries of the input production data.
production_time_resolution = 15 * 60  # 15min

production_modifiers = {summer_str: 20, winter_str: 1}
consumption_modifiers = {}

time_resolution = '15min'
cycle_counts_path = 'cycle_counts'

f_result_path = 'Results/{}/{}'

worker_count = multiprocessing.cpu_count()

