from datetime import datetime, timedelta

import java_classes
from const import date_format, one_second, time_zone
from java_classes import *


def advance_one_second(clock):
    now = Instant.now(clock)
    time = datetime.strptime(now.toString(), date_format)
    time = time + one_second
    return Clock(Instant.parse(time.strftime(date_format)), TimeZone.of(time_zone))


def add_channels(components):
    for component in components:
        ChannelUpdater.addChannels(component.channels())


def setup_simulation_java(start, main_config, support_config, grid_limit, min_energy):
    main = ManagedHybridEss()
    support = ManagedHybridEss()
    main.activate(None, main_config)
    support.activate(None, support_config)
    dummy_power = DummyPower()
    simulated_sum = Sum()
    clock = Clock(Instant.parse(start.strftime(date_format)), TimeZone.of(time_zone))
    component_manager = Component_Manager(clock)
    component_manager.addComponent(main)
    component_manager.addComponent(support)
    main._pythonBrideSetComponentManager(component_manager)
    support._pythonBrideSetComponentManager(component_manager)
    dummy_power.addEss(main)
    dummy_power.addEss(support)
    main._pyhtonBridgeSetPower(dummy_power)
    support._pyhtonBridgeSetPower(dummy_power)
    controller = HybridController("ctrl0", "HybridController", min_energy, grid_limit, 'ess0', 'ess1', simulated_sum,
                                  component_manager)
    add_channels([simulated_sum, component_manager, controller, main, support])
    cycle_worker = SimulatedCycleWorker(simulated_sum, dummy_power, main, support, controller, component_manager, clock)
    return cycle_worker


def setup_simulation(start, main_config, support_config, grid_limit, min_energy):
    main = ManagedHybridEss()
    support = ManagedHybridEss()
    main.activate(None, main_config)
    support.activate(None, support_config)
    dummy_power = DummyPower()
    simulated_sum = Sum()
    clock = Clock(Instant.parse(start.strftime(date_format)), TimeZone.of(time_zone))
    component_manager = Component_Manager(clock)
    component_manager.addComponent(main)
    component_manager.addComponent(support)
    main._pythonBrideSetComponentManager(component_manager)
    support._pythonBrideSetComponentManager(component_manager)
    dummy_power.addEss(main)
    dummy_power.addEss(support)
    main._pyhtonBridgeSetPower(dummy_power)
    support._pyhtonBridgeSetPower(dummy_power)
    controller = HybridController("ctrl0", "HybridController", min_energy, grid_limit, 'ess0', 'ess1', simulated_sum,
                                  component_manager)
    add_channels([simulated_sum, component_manager, controller, main, support])
    cycle_worker = CycleWorker(simulated_sum, dummy_power, main, support, controller, component_manager, clock)
    return cycle_worker


class CycleWorker():

    def __init__(self, sum, power, main, support, controller, component_manager, clock):
        self.openems_sum = sum
        self.power = power
        self.main = main
        self.support = support
        self.controller = controller
        self.clock = clock

    def update_clock(self):
        self.clock.leap(1, ChronoUnits.SECONDS)

    def get_datetime(self):
        now = Instant.now(self.clock)
        return datetime.strptime(now.toString(), date_format)

    def execute_cycle(self, production, consumption):
        self.openems_sum.setConsumption(consumption)
        self.openems_sum.setProduction(production)
        ChannelUpdater.updateChannels()
        self.main.handleEvent(after_process_image_event())
        self.support.handleEvent(after_process_image_event())
        self.controller.run()
        self.power.handleEvent(after_write_event())
        time_stamp = self.get_datetime()
        main_active_power = self.main.getActivePower().orElse(0)
        support_active_power = self.support.getActivePower().orElse(0)
        hess_active_power = main_active_power + support_active_power
        main_soc = self.main.getSoc().get()
        support_soc = self.main.getSoc().get()
        hess_soc = (main_soc + support_soc) / 2
        grid_active_power = - consumption + production + (main_active_power + support_active_power)
        self.update_clock()
        return [time_stamp, main_active_power, support_active_power, hess_active_power, main_soc, support_soc,
                hess_soc, grid_active_power, consumption, production]
