import jnius_config

jnius_config.set_classpath('.', 'openems-edge/jar/*')
from jnius import autoclass

Sum = autoclass('io.openems.edge.simulator.pythonBridge.SimulatedSum')
Component_Manager = autoclass('io.openems.edge.common.test.DummyComponentManager')
HybridController = autoclass('io.openems.edge.controller.ess.hybridess.controller.HybridControllerImpl')
ManagedHybridEss = autoclass('io.openems.edge.simulator.ess.symmetric.hybrid.EssSymmetricHybrid')
Config = autoclass('io.openems.edge.simulator.pythonBridge.EssConfig')
ChannelUpdater = autoclass('io.openems.edge.simulator.pythonBridge.ChannelUpdater')
DummyPower = autoclass('io.openems.edge.simulator.pythonBridge.SimulatedPower')
DummyConfigurationAdmin = autoclass('io.openems.edge.common.test.DummyConfigurationAdmin')
SimulatedCycleWorker = autoclass('io.openems.edge.simulator.pythonBridge.SimulatedCycleWorker')

Event = autoclass('org.osgi.service.event.Event')
Map = autoclass('java.util.HashMap')
after_process_image_str = "io/openems/edge/cycle/AFTER_PROCESS_IMAGE"
before_write_event_str = 'io/openems/edge/cycle/BEFORE_WRITE'
after_write_event_str = 'io/openems/edge/cycle/AFTER_WRITE'
Clock = autoclass('io.openems.edge.common.test.TimeLeapClock')
ChronoUnits = autoclass('java.time.temporal.ChronoUnit')
Instant = autoclass('java.time.Instant')
TimeZone = autoclass('java.time.ZoneId')


def after_process_image_event():
    return Event(after_process_image_str, Map())


def before_write_event():
    return Event(before_write_event_str, Map())


def after_write_event():
    return Event(after_write_event_str, Map())
