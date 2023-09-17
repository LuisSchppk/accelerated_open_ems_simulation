import pandas as pd
from matplotlib import pyplot as plt

from const import main_active_power_str, support_active_power_str, main_soc_str, support_soc_str, grid_str, \
    main_battery_activations_str, support_battery_activations_str, main_power_step_str, support_power_step_str, \
    production_str, consumption_str
from config import main_capacity, support_capacity


def plot_results(df, sim_id, result_path):

    # Active Power
    ax = df[[main_active_power_str, support_active_power_str]].plot(ylabel='Power in [W]', xlabel='')
    plt.tight_layout()
    plt.savefig(f'{result_path}/ActivePower_{sim_id}.pdf')
    plt.show()

    # SoC
    ax = df[[main_soc_str, support_soc_str]].plot(ylabel='SoC in [%]', xlabel='')
    # ax.axhline(y=20, xmin=0, xmax=1, c='red')
    # ax.axhline(y=22, xmin=0, xmax=1, c='red')
    # ax.axhline(y=50, xmin=0, xmax=1, c='orange')
    # ax.axhline(y=70, xmin=0, xmax=1, c='green')
    plt.tight_layout()
    plt.savefig(f'{result_path}/SoC_{sim_id}.pdf')
    plt.show()

    # Abs Stored power
    abs_energy = df[[main_soc_str, support_soc_str]]
    abs_energy.loc[:, main_soc_str] *= main_capacity
    abs_energy.loc[:, support_soc_str] *= support_capacity
    # abs_energy[main_soc_str] = abs_energy[[main_soc_str]] * main_capacity
    # abs_energy[support_soc_str] = abs_energy[[support_soc_str]] * support_capacity
    abs_energy = abs_energy.rename(columns={main_soc_str: 'Main Stored Energy', support_soc_str: 'Support Stored Energy'})

    ax = abs_energy.plot(ylabel='Stored Energy in Wh', xlabel='')
    plt.tight_layout()
    plt.savefig(f'{result_path}/abs_energy_{sim_id}.pdf')
    plt.show()

    # Grid power
    df[[grid_str]].plot(ylabel='Power in [W]', xlabel='')
    plt.tight_layout()
    plt.savefig(f'{result_path}/Grid_{sim_id}.pdf')
    plt.show()

    df[[consumption_str, production_str]].plot(ylabel='Power in [W]', xlabel='')
    plt.tight_layout()
    plt.savefig(f'{result_path}/production_consumption_{sim_id}.pdf')


def plot_charge_cycles(main_charge_cycles, support_charge_cycles, sim_id, cycles_path):
    ax = main_charge_cycles.plot(label='Main', ylabel='Number of Charge Cycles', xlabel='')
    support_charge_cycles.plot(ax=ax, label='Support', xlabel='')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'{cycles_path}/charge_cycles_{sim_id}.pdf')
    plt.show()


def plot_discharge_cycles(main_discharge_cycles, support_discharge_cycles, sim_id, cycles_path):
    if main_discharge_cycles.empty or support_discharge_cycles.empty:
        return  # Exit early on empty df to avoid error
    ax = main_discharge_cycles.plot(label='Main', ylabel='Number of Discharge Cycles', xlabel='')
    support_discharge_cycles.plot(ax=ax, label='Support', xlabel='')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'{cycles_path}/discharge_cycles_{sim_id}.pdf')
    plt.show()


def plot_additional_metrics(df, sim_id, result_path):
    df[[main_battery_activations_str, support_battery_activations_str]].plot(xlabel='', ylabel='Number of Activations')
    plt.tight_layout()
    plt.savefig(f'{result_path}/Activations_{sim_id}.pdf')
    plt.show()

    ax = df[[main_power_step_str, support_power_step_str]].boxplot(ylabel='Power Step in [W]')
    plt.tight_layout()
    plt.savefig(f'{result_path}/Power_Step_{sim_id}.pdf')
    plt.show()
