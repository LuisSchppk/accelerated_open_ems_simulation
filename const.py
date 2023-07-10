import os
from datetime import timedelta

main_active_power_str = 'Main Active Power [W]'
support_active_power_str = 'Support Active Power [W]'
hess_active_power_str = 'HESS Active Power [W]'
main_soc_str = 'Main SoC [%]'
support_soc_str = 'Support SoC [%]'
hess_soc_str = 'HESS SoC [%]'
consumption_str = 'Consumption [W]'
production_str = 'Production [W]'
grid_str = 'Grid [W]'
autarky_str = 'Autarky [%]'
self_consumption_str = 'Self-Consumption [W]'
date_time_str = 'Datetime'
main_power_step_str = 'Main Power Step [W/Cycle]'
support_power_step_str = 'Support Power Step [W/Cycle]'
main_battery_activations_str = 'Main Battery Activations'
support_battery_activations_str = 'Support Battery Activations'
cycle_count_str = 'cycle count'
main_charge_cycle_count_str = 'Main Charge Cycle Count'
main_discharge_cycle_count_str = 'Main Discharge Cycle Count'
support_charge_cycle_count_str = 'Support Charge Cycle Count'
support_discharge_cycle_count_str = 'Support Discharge Cycle Count'

cycle_special_metric_str = 'Barbaras Metric'
cycle_energy_str = 'total energy [Wh]'
f_cycle_duration_str = 'cycle_duration [{}]'
cycle_mean_power_str = 'mean power [W]'
cycle_start_str = 'start'
cycle_stop_str = 'stop'
cycle_median_power_str = 'median power [W]'

result_name_15m = 'base_metrics_15min_res'
result_name_1s = 'base_metrics_1s_res'

date_format = '%Y-%m-%dT%H:%M:%SZ'
one_second = timedelta(seconds=1)
time_zone = 'Europe/Paris'

low_grid_limit = 100_000
high_grid_limit = 200_000
default_min_energy = 50_000

main_capacity = 400_000
support_capacity = 220_000

consumption_time_resolution = 1 * 60  # 1min
production_time_resolution = 15 * 60  # 15min

production_modifier = 20
consumption_modifier = 1

time_resolution = '15min'
cycle_counts_path = 'cycle_counts'

df_columns = [date_time_str,
              main_active_power_str,
              support_active_power_str,
              hess_active_power_str,
              main_soc_str,
              support_soc_str,
              hess_soc_str,
              grid_str,
              consumption_str,
              production_str]
