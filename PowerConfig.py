from jnius import PythonJavaClass, autoclass, java_method

SolverStrategy = autoclass('io.openems.edge.ess.power.api.SolverStrategy')


class PowerConfig(PythonJavaClass):
    __javainterfaces__ = ['io.openems.edge.ess.core.power.Config']

    strategy_test = SolverStrategy.OPTIMIZE_BY_MOVING_TOWARDS_TARGET

    @java_method('()Z')
    def symmetricMode(self):
        return True

    @java_method('()io.openems.edge.ess.power.api.SolverStrategy')
    def strategy(self):
        return self.strategy_test


    @java_method('()Z')
    def debugMode(self):
        return True

    @java_method('()Z')
    def enablePid(self):
        return False

    @java_method('()D')
    def p(self):
        return 0.3

    @java_method('()D')
    def i(self):
        return 0.3

    @java_method('()D')
    def d(self):
        return 0.1

    @java_method('()java.lang.String')
    def webconsole_configurationFactory_nameHint(self):
        return 'ESS Power'


