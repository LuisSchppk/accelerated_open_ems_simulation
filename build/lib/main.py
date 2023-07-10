import cython
import jnius_config
jnius_config.set_classpath('.','*')
import jnius

from jnius import autoclass


if __name__ == '__main__':
    test = jnius.autoclass('OSGI-OPT.src.io.openems.edge.controller.ess.hybridess.controller.HybridControllerImpl')
    print(test.toString())