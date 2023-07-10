import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from apply_utils import mask_only_hess_sell_to_grid, get_hess_sells_surplus, autarky_hess, self_consumption
from const import *
from plotting import plot_results, plot_additional_metrics, plot_charge_cycles, plot_discharge_cycles
from simulation_data import SimulationData
from utils import interval_to_duration, append_to_csv, append_to_fastparquet, append_series, read_cycles

f_result_path = 'Results/{}'


def stats_autarky_and_self_consumption(results, interval, sim_id, result_path):
    mean_autarky = results[autarky_str].mean()
    median_autarky = results[autarky_str].median()
    mean_self_consumption = results[self_consumption_str].mean()
    median_self_consumption = results[self_consumption_str].median()

    # Theoretical maximum Autarky and SC
    interval_duration = interval_to_duration(interval)
    total_consumption = results[consumption_str].sum() * interval_duration
    total_production = results[production_str].sum() * interval_duration
    total_grid_energy = results[grid_str].sum() * interval_duration
    total_hess_charging = results[hess_active_power_str].loc[
                              results[hess_active_power_str] <= 0].sum() * -interval_duration

    # Autarky is maximized if all power from system was used
    max_autarky = 0
    max_sc = 100 if (total_consumption + total_hess_charging) >= total_production else (
                                                                                               total_consumption + total_hess_charging) / total_production
    if max_sc == 100:
        max_autarky = total_grid_energy / total_production  # all consumption could be covered by production
    else:
        max_autarky = ((total_consumption + total_hess_charging) - total_production) / (
                total_consumption + total_hess_charging)

    # Without trivial cases. No consumption and ess not charging => Autarky trivial 100%
    mask_hess = results[hess_active_power_str] >= 0
    mask_consumption = results[consumption_str] <= 0
    mask_untrivial_autarky = mask_hess & mask_consumption
    mask_untrivial_autarky = ~mask_untrivial_autarky
    untrivial_autarky = results.loc[mask_untrivial_autarky, [autarky_str]].squeeze()
    mean_untrivial_autarky = untrivial_autarky.mean()
    median_untrivial_autarky = untrivial_autarky.median()

    mask = results[production_str] == 0
    mask = ~mask
    untrivial_self_consumption = results.loc[mask, [self_consumption_str]].squeeze()
    mean_untrivial_self_consumption = untrivial_self_consumption.mean()
    median_untrivial_self_consumption = untrivial_self_consumption.median()

    output = os.linesep.join([f'Max Feasible Autarky and SC',
                              f'Max. Autarky {max_autarky} [%]',
                              f'Max. SC {max_sc} [%]',
                              'Metric with trivial cases:',
                              f'Mean Autarky [%] : {mean_autarky}',
                              f'Median Autarky [%] : {median_autarky}',
                              f'Mean Self Consumption [%] : {mean_self_consumption}',
                              f'Median Self Consumption [%] : {median_self_consumption}',
                              'Metric without trivial cases:',
                              f'Mean Autarky [%] : {mean_untrivial_autarky}',
                              f'Median Autarky [%] : {median_untrivial_autarky}',
                              f'Mean Self Consumption [%] : {mean_untrivial_self_consumption}',
                              f'Median Self Consumption [%] : {median_untrivial_self_consumption}'
                              ])
    with open(f'{result_path}/Autarky_and_Self_Consumption_{sim_id}.txt', 'w') as text_file:
        text_file.write(output)


def total_energies(grid_power, pv_power, hess_power, consumption, sim_id, interval, result_path):
    grid_consumption = grid_power.copy()
    sell_to_grid = grid_power.copy()
    pv_power = pv_power.copy()
    hess_discharge = hess_power.copy()
    hess_charge = hess_power.copy()
    consumption = consumption.copy()

    grid_consumption.loc[grid_consumption < 0] = 0
    sell_to_grid.loc[sell_to_grid > 0] = 0
    interval_duration = interval_to_duration(interval)
    hess_discharge.loc[hess_discharge < 0] = 0
    hess_charge.loc[hess_charge >= 0] = 0
    total_hess_charged = hess_charge.sum() * interval_duration
    total_hess_discharged = hess_discharge.sum() * interval_duration
    total_consumption = consumption.sum() * interval_duration

    df = pd.DataFrame({'grid_consumption': grid_consumption,
                       'sell_to_grid': sell_to_grid,
                       'pv_production': pv_power,
                       'hess_discharge': hess_discharge,
                       'consumption': consumption})

    only_hess_sells_to_grid = df.apply(mask_only_hess_sell_to_grid, axis=1)
    only_hess_sells_to_grid = df.loc[only_hess_sells_to_grid]
    only_hess_sells_to_grid = only_hess_sells_to_grid[['sell_to_grid']].squeeze()

    hess_sells_surplus = df.apply(get_hess_sells_surplus, axis=1)

    total_grid_energy_consumed = grid_consumption.sum() * interval_duration  # Wh
    total_energy_sold_to_grid = sell_to_grid.sum() * interval_duration  # Wh
    total_pv_produced = pv_power.sum() * interval_duration  # Wh

    # negative to be consistent with '-' => power is removed from system.
    total_hess_sold_to_grid = -(only_hess_sells_to_grid.sum() * interval_duration) \
                              - (hess_sells_surplus.sum() * interval_duration)
    total_pv_consumed = total_pv_produced + (total_energy_sold_to_grid - total_hess_sold_to_grid)
    output = f'Total Energy Consumed : {total_consumption} [Wh] {os.linesep}' \
             f'Total Grid Energy Consumed : {total_grid_energy_consumed} [Wh] {os.linesep}' \
             f'Total PV Energy Produced : {total_pv_produced} [Wh] {os.linesep}' \
             f'Total PV Energy Consumed : {total_pv_consumed} [Wh] {os.linesep}' \
             f'Total Energy Charged to HESS : {total_hess_charged} [Wh] {os.linesep}' \
             f'Total Energy Discharged by HESS : {total_hess_discharged} [Wh] {os.linesep}' \
             f'Total Energy Sold to Grid : {total_energy_sold_to_grid} [Wh] {os.linesep}' \
             f'Total HESS Energy Sold to Grid : {total_hess_sold_to_grid} [Wh] {os.linesep}' \
             f'Total PV Energy Sold To Grid : {-(total_pv_produced - total_pv_consumed)} [Wh]'
    with open(f'{result_path}/total_energy_consumed_{sim_id}.txt', 'w') as text_file:
        text_file.write(output)


def grid_limit_exceeded(grid_power, sim_id, result_path, grid_limit):

    limit_exceeded = grid_power.loc[abs(grid_power) > grid_limit]
    output = f'Total Number of Grid Limit exceeded: {limit_exceeded.size}'
    values = os.linesep.join(
        [f'{date_time_idx} : {power_value}' for (date_time_idx, power_value) in limit_exceeded.items()])
    output = os.linesep.join([output, values])
    with open(f'{result_path}/grid_limit_exceeded_{sim_id}.txt', 'w') as text_file:
        text_file.write(output)


def cleaned_power_step(active_power):
    mask = False
    tmp = active_power.diff()
    tmp[0] = 0
    mask2 = tmp != 0
    final_mask = mask | mask2
    cleaned_active_power = active_power.loc[final_mask]
    return cleaned_active_power.diff()


def battery_activations(active_power):
    def increase_count():
        nonlocal count
        count += 1
        return count

    count = 0
    mask_null = active_power == 0
    mask_start = active_power.diff(periods=-1) != 0
    mask_start = mask_start & mask_null

    battery_activations_count = pd.Series(data=mask_start)
    battery_activations_count = battery_activations_count.apply((lambda x: increase_count() if x else count))
    return battery_activations_count


def battery_cycles(active_power, date_time, interval, is_charge, p_count, total_capacity):
    def increase_count():
        nonlocal count
        count += 1
        return count

    count = p_count
    cycles = pd.DataFrame()

    # Duration of time resolution interval in [h]
    interval_duration = interval_to_duration(interval)

    # Unit of time resolution interval
    unit = re.sub('[0-9]', '', interval)
    filtered_power = active_power.copy()

    # Normalize charging to positive and remove active power belonging to other type of use.
    if is_charge:
        filtered_power[filtered_power > 0] = 0
        filtered_power = filtered_power * -1
    else:
        filtered_power[filtered_power < 0] = 0

    mask_null = filtered_power == 0
    mask_start = filtered_power.diff(periods=-1) < 0
    mask_stop = filtered_power.diff() < 0
    mask_start = mask_start & mask_null
    mask_stop = mask_stop & mask_null

    # Trivial Start if the examined interval began with charging/ discharging
    mask_start[0] = filtered_power[0] > 0
    if filtered_power[0] == 0 and filtered_power[1] > 0:
        mask_start[0] = True

    start_indices = np.where(mask_start)[0]
    stop_indices = np.where(mask_stop)[0]

    # Trivial Stop at the end of the examined interval.
    if start_indices.size > stop_indices.size:
        stop_indices = np.append(stop_indices, [filtered_power.size - 1])

    cycles[cycle_start_str] = start_indices
    cycles[cycle_stop_str] = stop_indices

    # Exit early if no cycles were detected. Not nice, but apply(...) fails for some reason on empty df.
    if cycles.empty:
        cycle_count = pd.DataFrame(data=np.full(active_power.size, count), index=date_time).squeeze()
        cycles = pd.DataFrame(columns=[cycle_start_str,
                                       cycle_stop_str,
                                       f_cycle_duration_str.format(unit),
                                       cycle_energy_str,
                                       cycle_mean_power_str,
                                       cycle_median_power_str,
                                       cycle_special_metric_str])
        return cycles, cycle_count

    # Calculate cycle metrics.
    cycles[f_cycle_duration_str.format(unit)] = (stop_indices - start_indices) * int((re.sub('[A-Za-z]', '', interval)))
    cycles[cycle_energy_str] = cycles[[cycle_start_str, cycle_stop_str]].apply(
        lambda x: (filtered_power.iloc[x[0]:x[1]].sum() * interval_duration),
        axis=1)
    cycles[cycle_mean_power_str] = cycles[[cycle_start_str, cycle_stop_str]].apply(
        lambda x: filtered_power.iloc[x[0]:x[1]].mean(),
        axis=1)
    cycles[cycle_median_power_str] = cycles[[cycle_start_str, cycle_stop_str]].apply(
        lambda x: filtered_power.iloc[x[0]:x[1]].median(),
        axis=1)
    cycles[cycle_special_metric_str] = cycles[cycle_energy_str].apply(lambda x: x / total_capacity)

    # Convert start from idx to datetime.
    cycles[cycle_start_str] = cycles[cycle_start_str].apply(lambda x: pd.to_datetime(date_time.iat[int(x)]))
    cycles[cycle_stop_str] = cycles[cycle_stop_str].apply(lambda x: date_time.iat[int(x)])

    # Calculate Cycle Count over date time for plot.
    cycle_count = pd.Series(data=mask_start)
    cycle_count = cycle_count.apply((lambda x: increase_count() if x else count))
    cycle_count = pd.DataFrame(data=cycle_count)
    cycle_count[date_time_str] = date_time
    cycle_count = cycle_count.set_index(date_time_str, drop=True)

    return cycles, cycle_count.squeeze()


def additional_1s_metrics(df, sim_id, result_path, simulation_data):
    result = df.copy()

    result[main_power_step_str] = cleaned_power_step(df[main_active_power_str])
    result[support_power_step_str] = cleaned_power_step(df[support_active_power_str])

    result[main_battery_activations_str] = battery_activations(df[[main_active_power_str]].squeeze())
    result[support_battery_activations_str] = battery_activations(df[[support_active_power_str]].squeeze())

    result[main_battery_activations_str] = result[main_battery_activations_str] \
                                           + simulation_data.get_value(main_battery_activations_str)
    result[support_battery_activations_str] = result[support_battery_activations_str] \
                                              + simulation_data.get_value(main_battery_activations_str)

    simulation_data.update_count(main_battery_activations_str, result[main_battery_activations_str].max())
    simulation_data.update_count(support_battery_activations_str, result[support_battery_activations_str].max())

    result = result.set_index(date_time_str, drop=True)
    filename = f'{result_path}/additional_metrics_{sim_id}.csv'
    append_to_csv(result, filename)


def calculate_cycle_metrics(main_charge, main_discharge, support_charge, support_discharge, sim_id,
                            interval, result_path):
    unit = re.sub('[0-9]', '', interval)
    charge_cycle_durations = append_series([main_charge[[f_cycle_duration_str.format(unit)]].squeeze(),
                                            support_charge[[f_cycle_duration_str.format(unit)]].squeeze()])
    charge_total_energy = append_series([main_charge[[cycle_energy_str]].squeeze(),
                                         support_charge[[cycle_energy_str]].squeeze()])
    charge_mean_power = append_series([main_charge[[cycle_mean_power_str]].squeeze(),
                                       support_charge[[cycle_mean_power_str]].squeeze()])
    charge_ratio_metric = append_series([main_charge[[cycle_special_metric_str]].squeeze(),
                                         support_charge[[cycle_special_metric_str]].squeeze()])

    discharge_cycle_durations = append_series([main_discharge[[f_cycle_duration_str.format(unit)]].squeeze(),
                                               support_discharge[[f_cycle_duration_str.format(unit)]].squeeze()])
    discharge_total_energy = append_series([main_discharge[[cycle_energy_str]].squeeze(),
                                            support_discharge[[cycle_energy_str]].squeeze()])
    discharge_mean_power = append_series([main_discharge[[cycle_mean_power_str]].squeeze(),
                                          support_discharge[[cycle_mean_power_str]].squeeze()])
    discharge_ratio_metric = append_series([main_discharge[[cycle_special_metric_str]].squeeze(),
                                            support_discharge[[cycle_special_metric_str]].squeeze()])

    overall_cycle_durations = pd.concat([charge_cycle_durations, discharge_cycle_durations])
    overall_total_energy = pd.concat([charge_total_energy, discharge_total_energy])
    tmp = abs(charge_mean_power.copy())
    overall_mean_power = pd.concat([tmp, discharge_mean_power])
    overall_ratio_metric = pd.concat([charge_ratio_metric, discharge_ratio_metric])

    avg_charge_cycle_duration = charge_cycle_durations.mean().squeeze()
    avg_charge_total_energy = charge_total_energy.mean().squeeze()
    avg_charge_mean_power = charge_mean_power.mean().squeeze()
    avg_charge_ratio_metric = charge_ratio_metric.mean().squeeze()

    avg_discharge_cycle_duration = discharge_cycle_durations.mean().squeeze()
    avg_discharge_total_energy = discharge_total_energy.mean().squeeze()
    avg_discharge_mean_power = discharge_mean_power.mean().squeeze()
    avg_discharge_ratio_metric = discharge_ratio_metric.mean().squeeze()

    avg_overall_cycle_duration = overall_cycle_durations.mean().squeeze()
    avg_overall_total_energy = overall_total_energy.mean().squeeze()
    avg_overall_mean_power = overall_mean_power.mean().squeeze()
    avg_overall_ratio_metric = overall_ratio_metric.mean().squeeze()

    output = os.linesep.join(['Charge',
                              f'Average Charge Cycle Duration: {avg_charge_cycle_duration}',
                              f'Average Charge Energy [Wh] : {avg_charge_total_energy}',
                              f'Average Charge Mean Power [W]: {avg_charge_mean_power}',
                              f'Average Charge Ratio Metric : {avg_charge_ratio_metric}',
                              'Discharge',
                              f'Average Discharge Cycle Duration: {avg_discharge_cycle_duration}',
                              f'Average Discharge Energy [Wh] : {avg_discharge_total_energy}',
                              f'Average Discharge Mean Power [W]: {avg_discharge_mean_power}',
                              f'Average Discharge Ratio Metric : {avg_discharge_ratio_metric}',
                              'Overall',
                              f'Average Overall Cycle Duration: {avg_overall_cycle_duration}',
                              f'Average Overall Energy [Wh] : {avg_overall_total_energy}',
                              f'Average Overall Mean Power [W]: {avg_overall_mean_power}',
                              f'Average Overall Ratio Metric : {avg_overall_ratio_metric}',
                              ])
    with open(f'{result_path}/cycle_metrics_{sim_id}_{interval}.txt', 'w') as text_file:
        text_file.write(output)



def calculate_and_store_cycles(results, cycles_path, sim_id, interval, simulation_data):
    main_charge_cycles, main_charge_cycle_count = \
        battery_cycles(results[[main_active_power_str]].squeeze(),
                       results[[date_time_str]].squeeze(),
                       interval,
                       True,
                       simulation_data.get_value(main_charge_cycle_count_str),
                       main_capacity)
    main_discharge_cycles, main_discharge_cycle_count = \
        battery_cycles(results[[main_active_power_str]].squeeze(),
                       results[[date_time_str]].squeeze(),
                       interval,
                       False,
                       simulation_data.get_value(
                           main_discharge_cycle_count_str),
                       main_capacity)

    support_charge_cycles, support_charge_cycle_count = \
        battery_cycles(results[[support_active_power_str]].squeeze(),
                       results[[date_time_str]].squeeze(),
                       interval,
                       True,
                       simulation_data.get_value(
                           support_charge_cycle_count_str),
                       support_capacity)
    support_discharge_cycles, support_discharge_cycle_count = \
        battery_cycles(results[[support_active_power_str]].squeeze(),
                       results[[date_time_str]].squeeze(),
                       interval,
                       False,
                       simulation_data.get_value(support_discharge_cycle_count_str),
                       support_capacity)

    cycle_counts = pd.DataFrame({main_charge_cycle_count_str: main_charge_cycle_count,
                                 main_discharge_cycle_count_str: main_discharge_cycle_count,
                                 support_charge_cycle_count_str: support_charge_cycle_count,
                                 support_discharge_cycle_count_str: support_discharge_cycle_count})

    simulation_data.update_count(main_charge_cycle_count_str, main_charge_cycle_count.max())
    simulation_data.update_count(main_discharge_cycle_count_str, main_discharge_cycle_count.max())
    simulation_data.update_count(support_charge_cycle_count_str, support_charge_cycle_count.max())
    simulation_data.update_count(support_discharge_cycle_count_str, support_discharge_cycle_count.max())

    append_to_csv(main_charge_cycles, f'{cycles_path}/main_charge_{sim_id}.csv', False)
    append_to_csv(main_discharge_cycles, f'{cycles_path}/main_discharge_{sim_id}.csv', False)
    append_to_csv(support_charge_cycles, f'{cycles_path}/support_charge_{sim_id}.csv', False)
    append_to_csv(support_discharge_cycles, f'{cycles_path}/support_discharge_{sim_id}.csv', False)
    append_to_csv(cycle_counts, f'{cycles_path}/cycle_counts_{sim_id}.csv')

    return main_charge_cycles, main_discharge_cycles, support_charge_cycles, support_discharge_cycles, cycle_counts


def evaluate_and_store_on_the_run(results, sim_id, simulation_data):
    print('Chunk')
    result_path = f_result_path.format(sim_id)
    cycles_path = f'{result_path}/{cycle_counts_path}'
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(cycles_path, exist_ok=True)

    results = pd.DataFrame(data=results, columns=df_columns)

    results[date_time_str] = pd.to_datetime(results[date_time_str], unit='s')

    append_to_fastparquet(results, f'{result_path}/{result_name_1s}.parquet')

    # 1s metrics
    additional_1s_metrics(results, sim_id, result_path, simulation_data)
    calculate_and_store_cycles(results, cycles_path, sim_id, '1s', simulation_data)

    # time_resolution (15m) metrics
    results = results.set_index(date_time_str, drop=True)
    results = results.resample(time_resolution).mean()
    results[autarky_str] = results.apply(autarky_hess, axis=1)
    results[self_consumption_str] = results.apply(self_consumption, axis=1)
    append_to_csv(results, f'{result_path}/{result_name_15m}.csv')


def evaluate_and_store_final(sim_id, grid_limit):
    result_path = f_result_path.format(sim_id)
    results = pd.read_csv(f'{result_path}/{result_name_15m}.csv', index_col=date_time_str)
    cycles_path = f'{result_path}/{cycle_counts_path}'

    main_charge_cycles, main_discharge_cycles, support_charge_cycles, support_discharge_cycles, cycle_counts \
        = read_cycles(cycles_path, sim_id)
    additional_metrics = pd.read_csv(f'{result_path}/additional_metrics_{sim_id}.csv', index_col=date_time_str)

    stats_autarky_and_self_consumption(results=results, result_path=result_path, sim_id=sim_id, interval='15m')
    total_energies(grid_power=results[grid_str].squeeze(), pv_power=results[production_str].squeeze(),
                   hess_power=results[hess_active_power_str].squeeze(), consumption=results[consumption_str].squeeze(),
                   result_path=result_path, interval='15m', sim_id=sim_id)
    grid_limit_exceeded(results[grid_str].squeeze(), result_path=result_path, sim_id=sim_id, grid_limit=grid_limit)
    plot_results(df=results.copy(), result_path=result_path, sim_id=sim_id)

    calculate_cycle_metrics(main_charge_cycles, main_discharge_cycles, support_charge_cycles, support_discharge_cycles,
                            sim_id,'1s', cycles_path)

    plot_additional_metrics(additional_metrics, sim_id, result_path)
    plot_charge_cycles(cycle_counts[main_charge_cycle_count_str].squeeze(),
                       cycle_counts[support_charge_cycle_count_str].squeeze(),
                       sim_id, cycles_path)
    plot_discharge_cycles(cycle_counts[main_discharge_cycle_count_str].squeeze(),
                          cycle_counts[support_discharge_cycle_count_str].squeeze(), sim_id,
                          cycles_path)
