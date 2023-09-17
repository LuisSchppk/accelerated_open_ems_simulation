import multiprocessing
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
main_soc_state_str = "Main SoC State"
support_soc_state_str = "Support SoC State"

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

df_columns = [date_time_str,
              main_active_power_str,
              support_active_power_str,
              hess_active_power_str,
              main_soc_str,
              support_soc_str,
              hess_soc_str,
              grid_str,
              consumption_str,
              production_str,
              main_soc_state_str,
              support_soc_state_str]