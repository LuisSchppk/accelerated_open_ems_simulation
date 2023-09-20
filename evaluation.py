import os
import re

import numpy as np
import pandas as pd

from apply_utils import mask_only_hess_sell_to_grid, get_hess_sells_surplus, autarky_hess, self_consumption
from config import f_result_path, cycle_counts_path, support_capacity, main_capacity, time_resolution
from const import *
from plotting import plot_results, plot_additional_metrics, plot_charge_cycles, plot_discharge_cycles
from utils import interval_to_duration, append_to_csv, append_to_fastparquet, append_series, read_cycles, \
    id_to_site_and_season_grid


def stats_autarky_and_self_consumption(results, interval, sim_id, result_path):
    """
    Calculation of avg. mean, and max. Autarky and Selfconsumption.
    :param results: pd.Dataframe containing the final results.
    :param interval: time resolution of the results.
    :param sim_id: Simulation ID to name result files according to simulation and scenario.
    :param result_path: file path to store results.
    :return:
    """
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

    # Calculation for theoretical maximum autarky is not correct. I could not come up with the correct one.
    max_autarky = 0
    max_sc = 100 if (total_consumption + total_hess_charging) >= total_production \
                 else (total_consumption + total_hess_charging) / total_production
    if max_sc == 100:
        max_autarky = total_grid_energy / total_production  # all consumption could be covered by production
    else:
        max_autarky = ((total_consumption + total_hess_charging) - total_production) / (
                total_consumption + total_hess_charging)

    # Without trivial cases. No consumption and ess not charging => Autarky is trivially 100%
    mask_hess = results[hess_active_power_str] >= 0 # Not charging
    mask_consumption = results[consumption_str] <= 0 # No Consumption
    mask_untrivial_autarky = mask_hess & mask_consumption
    mask_untrivial_autarky = ~mask_untrivial_autarky
    untrivial_autarky = results.loc[mask_untrivial_autarky, [autarky_str]]

    mean_untrivial_autarky = untrivial_autarky.mean()
    median_untrivial_autarky = untrivial_autarky.median()

    # Make sure result is not a pd.Series containing one entry.
    if isinstance(mean_untrivial_autarky, pd.Series):
        mean_untrivial_autarky = mean_untrivial_autarky.squeeze()

    if isinstance(median_untrivial_autarky, pd.Series):
        median_untrivial_autarky = median_untrivial_autarky.squeeze()

    # Without trivial cases for SC, i.e., no production.
    mask = results[production_str] == 0
    mask = ~mask
    untrivial_self_consumption = results.loc[mask, [self_consumption_str]]

    if not isinstance(untrivial_self_consumption, np.int64):
        mean_untrivial_self_consumption = untrivial_self_consumption.mean()
        median_untrivial_self_consumption = untrivial_self_consumption.median()
    else:
        mean_untrivial_self_consumption = untrivial_self_consumption
        median_untrivial_self_consumption = untrivial_self_consumption

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


def total_energies(grid_power, pv_power, hess_power, main_power, support_power, consumption, sim_id, interval,
                   result_path):
    """
    Calculate and store information about the total energy produced, consumed, charged and discharged.
    :param grid_power: Time series of power drawn and discharged to grid in [W].
    :param pv_power: Time series of power generated by PV in [W].
    :param hess_power: Time series of power charged and discharged by the whole HESS in [W].
    :param main_power: Time series of power charged and discharged by the MAIN ESS in [W].
    :param support_power: Time series of power charged and discharged by the SUPPORT ESS in [W].
    :param consumption: Time series of power consumed by the load in [W].
    :param sim_id: Simulation ID to name result files according to simulation and scenario.
    :param interval: String <Amount><Unit> denoting the time delta between two entries. Unit either 'h' 'm' or 's'.
    :param result_path: file path to store the results.
    :return:
    """
    grid_consumption = grid_power.copy()
    sell_to_grid = grid_power.copy()
    pv_power = pv_power.copy()
    consumption = consumption.copy()

    hess_charge = hess_power.copy()
    hess_discharge = hess_power.copy()

    main_charge = main_power.copy()
    main_discharge = main_power.copy()

    support_charge = support_power.copy()
    support_discharge = support_power.copy()

    interval_duration = interval_to_duration(interval)

    # Separate data sets in to positive and negative power.
    grid_consumption.loc[grid_consumption < 0] = 0
    sell_to_grid.loc[sell_to_grid > 0] = 0

    hess_charge.loc[hess_charge >= 0] = 0
    hess_discharge.loc[hess_discharge < 0] = 0

    main_charge.loc[main_charge >= 0] = 0
    main_discharge.loc[main_discharge < 0] = 0

    support_charge.loc[support_charge >= 0] = 0
    support_discharge.loc[support_discharge < 0] = 0

    # Calculate Energies by power * time.
    total_hess_charged = hess_charge.sum() * interval_duration
    total_hess_discharged = hess_discharge.sum() * interval_duration

    total_main_charged = main_charge.sum() * interval_duration
    total_main_discharged = main_discharge.sum() * interval_duration

    total_support_charged = support_charge.sum() * interval_duration
    total_support_discharged = support_discharge.sum() * interval_duration

    total_consumption = consumption.sum() * interval_duration

    total_grid_energy_consumed = grid_consumption.sum() * interval_duration  # Wh
    total_energy_sold_to_grid = sell_to_grid.sum() * interval_duration  # Wh
    total_pv_produced = pv_power.sum() * interval_duration  # Wh

    # Build df from single series.
    df = pd.DataFrame({'grid_consumption': grid_consumption,
                       'sell_to_grid': sell_to_grid,
                       'pv_production': pv_power,
                       'hess_discharge': hess_discharge,
                       'consumption': consumption})

    # Calculate times, where only the hess sells to grid
    only_hess_sells_to_grid = df.apply(mask_only_hess_sell_to_grid, axis=1)
    only_hess_sells_to_grid = df.loc[only_hess_sells_to_grid]
    only_hess_sells_to_grid = only_hess_sells_to_grid[['sell_to_grid']].squeeze()

    hess_sells_surplus = df.apply(get_hess_sells_surplus, axis=1)

    # negative to be consistent with '-' => power is removed from system.
    total_hess_sold_to_grid = -(only_hess_sells_to_grid.sum() * interval_duration) \
                              - (hess_sells_surplus.sum() * interval_duration)

    total_pv_consumed = total_pv_produced + (total_energy_sold_to_grid - total_hess_sold_to_grid)

    output = f'Total Energy Consumed : {total_consumption} [Wh] {os.linesep}' \
             f'Total PV Energy Produced : {total_pv_produced} [Wh] {os.linesep}' \
             f'Total PV Energy Consumed : {total_pv_consumed} [Wh] {os.linesep}' \
             f'Total Grid Energy Consumed : {total_grid_energy_consumed} [Wh] {os.linesep}' \
             f'Batteries: {os.linesep}' \
             f'Total Energy Charged to HESS : {total_hess_charged} [Wh] {os.linesep}' \
             f'Total Energy Discharged by HESS : {total_hess_discharged} [Wh] {os.linesep}' \
             f'Total Energy Charged to MAIN : {total_main_charged} [Wh] {os.linesep}' \
             f'Total Energy Discharged by MAIN : {total_main_discharged} [Wh] {os.linesep}' \
             f'Total Energy Charged to SUPPORT : {total_support_charged} [Wh] {os.linesep}' \
             f'Total Energy Discharged by SUPPORT : {total_support_discharged} [Wh] {os.linesep}' \
             f'Sell to Grid: {os.linesep}' \
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


def calculate_power_step(active_power):
    """
    Calculate the power step, i.e., the difference of one power value to the next power value.

    :param active_power: pd.Series containing the power values.
    :return: pd.Series containing the power step values.
    """
    mask = False
    tmp = active_power.diff()
    tmp[0] = 0
    mask2 = tmp != 0
    final_mask = mask | mask2
    cleaned_active_power = active_power.loc[final_mask]
    return cleaned_active_power.diff()


def battery_activations(active_power, p_count):
    """
    Calculate the battery activations. An activation is any time point where to power goes from 0 to some value != 0.
    :param active_power: pd.Series containing the power values.
    :param p_count: Amount of activations passed from last chunk by simulation_data.
    :return:
    """
    def increase_count():
        nonlocal count
        count += 1
        return count

    count = p_count
    mask_null = active_power == 0
    mask_start = active_power.diff(periods=-1) != 0
    mask_start = mask_start & mask_null

    battery_activations_count = pd.Series(data=mask_start)
    battery_activations_count = battery_activations_count.apply((lambda x: increase_count() if x else count))
    return battery_activations_count


def battery_cycles(active_power, date_time_index, interval, is_charge, p_count, total_capacity):
    """
    Calculates a Charge and Discharge Battery Cycle for one batterie.
    One Battery Cycle is the Time between a time step where power = 0 to where
    power = 0 again or the sign changes.

    :param active_power: pd.Series containing the power values.
    :param date_time_index: pd.Series containing the date_time_index of the result df.
    :param interval: time resolution of the results.
    :param is_charge: Flag whether the charge or discharge cycles are examined.
    :param p_count: Amount of activations passed from last chunk by simulation_data.
    :param total_capacity: Capacity of the Battery in [Wh].
    :return:
    """
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

    if is_charge:
        # Normalize charging to positive and remove active power belonging to other type of use.
        filtered_power[filtered_power > 0] = 0
        filtered_power = filtered_power * -1
    else:
        #  Set all entries where the battery charges to zero.
        filtered_power[filtered_power < 0] = 0

    # Boolean mask: entry is zero.
    mask_null = filtered_power == 0

    # Boolean mask: Power increased.
    mask_start = filtered_power.diff(periods=-1) < 0

    # Boolean mask: Power decreased.
    mask_stop = filtered_power.diff() < 0

    # Boolean mask: Power increased from 0 to some value != 0.
    mask_start = mask_start & mask_null

    # Boolean mask: Power decreased from some value != 0 to 0
    mask_stop = mask_stop & mask_null

    # Trivial Start if the examined interval began with charging/ discharging.
    mask_start[0] = filtered_power[0] > 0
    if filtered_power[0] == 0 and filtered_power[1] > 0:
        mask_start[0] = True

    # Convert Boolean Masks to array containing the indices where mask is true.
    start_indices = np.where(mask_start)[0]
    stop_indices = np.where(mask_stop)[0]

    # Trivial Stop at the end of the examined interval.
    if start_indices.size > stop_indices.size:
        stop_indices = np.append(stop_indices, [filtered_power.size - 1])

    # Add Column with all Start indices and Column with the respective stop indices.
    cycles[cycle_start_str] = start_indices
    cycles[cycle_stop_str] = stop_indices

    # Exit early if no cycles were detected. Not nice, but apply(...) fails for some reason on empty df.
    if cycles.empty:
        # create as df not series for index
        cycle_count = pd.DataFrame(data=np.full(active_power.size, count), index=date_time_index).squeeze()
        cycles = pd.DataFrame(columns=[cycle_start_str,
                                       cycle_stop_str,
                                       f_cycle_duration_str.format(unit),
                                       cycle_energy_str,
                                       cycle_mean_power_str,
                                       cycle_median_power_str,
                                       cycle_special_metric_str])
        return cycles, cycle_count

    # Calculate cycle metrics.

    # Duration of the Cycle.
    cycles[f_cycle_duration_str.format(unit)] = (stop_indices - start_indices) * int((re.sub('[A-Za-z]', '', interval)))

    # Total Energy in [Wh] Charged or discharged during the cycle.
    cycles[cycle_energy_str] = cycles[[cycle_start_str, cycle_stop_str]].apply(
        lambda x: (filtered_power.iloc[x[0]:x[1]].sum() * interval_duration),
        axis=1)

    # Mean Power [W] charged with or discharged with during the cycle.
    cycles[cycle_mean_power_str] = cycles[[cycle_start_str, cycle_stop_str]].apply(
        lambda x: filtered_power.iloc[x[0]:x[1]].mean(),
        axis=1)

    # Median Power [W] charged with or discharged with during the cycle.
    cycles[cycle_median_power_str] = cycles[[cycle_start_str, cycle_stop_str]].apply(
        lambda x: filtered_power.iloc[x[0]:x[1]].median(),
        axis=1)

    # 'Special Metric' (We do not know the correct name) Energy [Wh] divided by Total Capacity [Wh] of the Battery.
    cycles[cycle_special_metric_str] = cycles[cycle_energy_str].apply(lambda x: x / total_capacity)

    # Convert start from idx to datetime.
    cycles[cycle_start_str] = cycles[cycle_start_str].apply(lambda x: pd.to_datetime(date_time_index.iat[int(x)]))
    cycles[cycle_stop_str] = cycles[cycle_stop_str].apply(lambda x: date_time_index.iat[int(x)])

    # Calculate Cycle Count over date time for plot.
    cycle_count = pd.Series(data=mask_start)
    cycle_count = cycle_count.apply((lambda x: increase_count() if x else count))
    cycle_count = pd.DataFrame(data=cycle_count)
    cycle_count[date_time_str] = date_time_index
    cycle_count = cycle_count.set_index(date_time_str, drop=True)

    return cycles, cycle_count.squeeze()


def additional_1s_metrics(df, sim_id, result_path, simulation_data):
    """
    Calculates the metrics 'main_power_step', 'support_power_step'
    and 'main_battery_activations', 'support_battery_activations'.

    Both metrics have not been really used to estimate the performance of the ESS System. Power Step helped me to
    identify some bugs.
    I think the code for Battery Activations might be bugged. The code for the Cycle metrics is more up to date in this
    regard.

    :param df: pd.Dataframe containing the base results.
    :param sim_id: Simulation ID to name result files according to simulation and scenario.
    :param result_path: Filepath to location where results should be stored.
    :param simulation_data: DSO containing values which need to be persistent across chunks.
    :return:
    """
    result = df.copy()

    result[main_power_step_str] = calculate_power_step(df[main_active_power_str])
    result[support_power_step_str] = calculate_power_step(df[support_active_power_str])

    result[main_battery_activations_str] = battery_activations(df[[main_active_power_str]].squeeze(),
                                                               simulation_data.get_value(main_battery_activations_str))
    result[support_battery_activations_str] = battery_activations(df[[support_active_power_str]].squeeze(),
                                                                  simulation_data.get_value(
                                                                      support_battery_activations_str))

    tmp = result[main_battery_activations_str].max()
    tmp2 = result[support_battery_activations_str].max()
    simulation_data.update_count(main_battery_activations_str, result[main_battery_activations_str].max())
    simulation_data.update_count(support_battery_activations_str, result[support_battery_activations_str].max())

    result = result.set_index(date_time_str, drop=True)
    filename = f'{result_path}/additional_metrics_{sim_id}.csv'
    append_to_csv(result, filename)


def cycle_metric_helper(durations, energies, powers, ratios):
    avg_cycle_duration = durations.mean()
    avg_total_energy = energies.mean()
    avg_mean_power = powers.mean()
    avg_ratio_metric = ratios.mean()

    if isinstance(avg_cycle_duration, pd.Series):
        avg_cycle_duration = avg_cycle_duration.squeeze()
    if isinstance(avg_total_energy, pd.Series):
        avg_total_energy = avg_total_energy.squeeze()
    if isinstance(avg_total_energy, pd.Series):
        avg_mean_power = avg_mean_power.squeeze()
    if isinstance(avg_total_energy, pd.Series):
        avg_ratio_metric = avg_ratio_metric.squeeze()
    return avg_cycle_duration, avg_total_energy, avg_mean_power, avg_ratio_metric


def calculate_cycle_metrics(main_charge, main_discharge, support_charge, support_discharge, sim_id,
                            interval, result_path):

    # get time unit of interval 'h', 'm' or 's'
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

    avg_charge_cycle_duration, avg_charge_total_energy, avg_charge_mean_power, avg_charge_ratio_metric \
        = cycle_metric_helper(charge_cycle_durations, charge_total_energy, charge_mean_power, charge_ratio_metric)

    avg_discharge_cycle_duration, avg_discharge_total_energy, avg_discharge_mean_power, avg_discharge_ratio_metric \
        = cycle_metric_helper(discharge_cycle_durations, discharge_total_energy, discharge_mean_power, discharge_ratio_metric)

    avg_overall_cycle_duration, avg_overall_total_energy, avg_overall_mean_power, avg_overall_ratio_metric \
        = cycle_metric_helper(overall_cycle_durations, overall_total_energy, overall_mean_power, overall_ratio_metric)

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
    """
    Calculates the Charge and Discharge Battery Cycle metrics for all batteries.
    One Battery Cycle is the Time between a time step where power = 0 to where
    power = 0 again or the sign changes.

    :param results: pd.Dataframe containing the results
    :param cycles_path: Filepath to store the results.
    :param sim_id: Simulation ID to name result files according to simulation and scenario.
    :param interval: time resolution of the results.
    :param simulation_data: DSO containing values which need to be persistent across chunks.
    :return:
    """
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

    # Build one DF from the single results.
    cycle_counts = pd.DataFrame({main_charge_cycle_count_str: main_charge_cycle_count,
                                 main_discharge_cycle_count_str: main_discharge_cycle_count,
                                 support_charge_cycle_count_str: support_charge_cycle_count,
                                 support_discharge_cycle_count_str: support_discharge_cycle_count})

    # Update the persistent DSO for the next Chunk.
    simulation_data.update_count(main_charge_cycle_count_str, main_charge_cycle_count.max())
    simulation_data.update_count(main_discharge_cycle_count_str, main_discharge_cycle_count.max())
    simulation_data.update_count(support_charge_cycle_count_str, support_charge_cycle_count.max())
    simulation_data.update_count(support_discharge_cycle_count_str, support_discharge_cycle_count.max())

    # Store intermediary results as csv.
    # TODO: With time resolution of 1s it might be faster to convert this to fastparquet as well.
    append_to_csv(main_charge_cycles, f'{cycles_path}/main_charge_{sim_id}.csv', False)
    append_to_csv(main_discharge_cycles, f'{cycles_path}/main_discharge_{sim_id}.csv', False)
    append_to_csv(support_charge_cycles, f'{cycles_path}/support_charge_{sim_id}.csv', False)
    append_to_csv(support_discharge_cycles, f'{cycles_path}/support_discharge_{sim_id}.csv', False)
    append_to_csv(cycle_counts, f'{cycles_path}/cycle_counts_{sim_id}.csv')

    return main_charge_cycles, main_discharge_cycles, support_charge_cycles, support_discharge_cycles, cycle_counts


def evaluate_and_store_intermediary_results(results, sim_id, simulation_data):

    # Construct result path from simulation id
    site, season_grid = id_to_site_and_season_grid(sim_id)
    result_path = f_result_path.format(site, season_grid)
    cycles_path = f'{result_path}/{cycle_counts_path}'

    # Create directories for results.
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(cycles_path, exist_ok=True)

    # convert np.ndarray of results to pd.Dataframe.
    results = pd.DataFrame(data=results, columns=df_columns)

    # Convert int date time to pd.datetime.
    results[date_time_str] = pd.to_datetime(results[date_time_str], unit='s')

    # Append the base results to the parquet file of previous intermediary results of this simulation.
    append_to_fastparquet(results, f'{result_path}/{result_name_1s}.parquet')

    # Calculate the extended results. which have a time resolution of [1s]
    additional_1s_metrics(results, sim_id, result_path, simulation_data)
    calculate_and_store_cycles(results, cycles_path, sim_id, '1s', simulation_data)

    # Down sample time_resolution for base results from 1s to 15m
    results = results.set_index(date_time_str, drop=True)
    results = results.resample(time_resolution).mean()

    # Calculate the metrics, which have a 15min time resolution. Autarky and Self Consumption.
    results[autarky_str] = results.apply(autarky_hess, axis=1)
    results[self_consumption_str] = results.apply(self_consumption, axis=1)
    append_to_csv(results, f'{result_path}/{result_name_15m}.csv')


def evaluate_and_store_final(sim_id, grid_limit):

    # Construct result path from simulation id
    site, season_grid = id_to_site_and_season_grid(sim_id)
    result_path = f_result_path.format(site, season_grid)
    results = pd.read_csv(f'{result_path}/{result_name_15m}.csv', index_col=date_time_str)
    results.index = pd.to_datetime(results.index)
    cycles_path = f'{result_path}/{cycle_counts_path}'

    # Read the total intermediary results to evaluate the final results.
    main_charge_cycles, main_discharge_cycles, support_charge_cycles, support_discharge_cycles, cycle_counts \
        = read_cycles(cycles_path, sim_id)
    additional_metrics = pd.read_csv(f'{result_path}/additional_metrics_{sim_id}.csv', index_col=date_time_str)
    additional_metrics.index = pd.to_datetime(additional_metrics.index)

    # Calculate final metrics
    stats_autarky_and_self_consumption(results=results, result_path=result_path, sim_id=sim_id, interval='15m')
    total_energies(grid_power=results[grid_str].squeeze(), pv_power=results[production_str].squeeze(),
                   hess_power=results[hess_active_power_str].squeeze(),
                   main_power=results[main_active_power_str].squeeze(),
                   support_power=results[support_active_power_str].squeeze(),
                   consumption=results[consumption_str].squeeze(),
                   result_path=result_path, interval='15m', sim_id=sim_id)
    grid_limit_exceeded(results[grid_str].squeeze(), result_path=result_path, sim_id=sim_id, grid_limit=grid_limit)
    plot_results(df=results.copy(), result_path=result_path, sim_id=sim_id)

    calculate_cycle_metrics(main_charge_cycles, main_discharge_cycles, support_charge_cycles, support_discharge_cycles,
                            sim_id, '1s', cycles_path)

    plot_additional_metrics(additional_metrics, sim_id, result_path)
    plot_charge_cycles(cycle_counts[main_charge_cycle_count_str].squeeze(),
                       cycle_counts[support_charge_cycle_count_str].squeeze(),
                       sim_id, cycles_path)
    plot_discharge_cycles(cycle_counts[main_discharge_cycle_count_str].squeeze(),
                          cycle_counts[support_discharge_cycle_count_str].squeeze(), sim_id,
                          cycles_path)
