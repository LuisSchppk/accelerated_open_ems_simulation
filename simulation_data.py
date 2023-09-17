from const import main_charge_cycle_count_str, support_charge_cycle_count_str, main_discharge_cycle_count_str, \
    support_discharge_cycle_count_str, main_battery_activations_str, support_battery_activations_str


class SimulationData:
    """
    Class for Data Storage Object used to store certain values across different chunks of the simulation.
    """
    def __init__(self):
        self.charge_cycle_count = {main_charge_cycle_count_str: 0,
                                   support_charge_cycle_count_str: 0}
        self.discharge_cycle_count = {main_discharge_cycle_count_str: 0,
                                      support_discharge_cycle_count_str: 0}
        self.activations = {main_battery_activations_str: 0,
                            support_battery_activations_str: 0}

    def update_count(self, key, value):
        if key is main_charge_cycle_count_str or key is support_charge_cycle_count_str:
            self.charge_cycle_count.update({key: value})
        elif key is main_discharge_cycle_count_str or key is support_discharge_cycle_count_str:
            self.discharge_cycle_count.update({key: value})
        elif key is main_battery_activations_str or support_battery_activations_str:
            self.activations.update({key: value})
        else:
            raise Exception(f'Key {key} not found for updating data.')

    def get_value(self, key):
        value = 0
        if key is main_charge_cycle_count_str or key is support_charge_cycle_count_str:
            value = self.charge_cycle_count.get(key)
        elif key is main_discharge_cycle_count_str or key is support_discharge_cycle_count_str:
            value = self.discharge_cycle_count.get(key)
        elif key is main_battery_activations_str or support_battery_activations_str:
            value = self.activations.get(key)
        else:
            raise Exception(f'Key {key} not found for getting data.')
        return value
