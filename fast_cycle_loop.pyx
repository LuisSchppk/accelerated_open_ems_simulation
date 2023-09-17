
import numpy as np
cimport numpy as np

from evaluation import evaluate_and_store_intermediary_results
from simulation_data import SimulationData

# This file contains Cython code. The code has to be compiled to C. (PyCharm: Tools-> run setup.py -> build_ext)

cdef c_fast_cycle_loop_java(int p_simulation_time, int p_column_count,
                            int p_production_time_resolution, int p_consumption_time_resolution,
                            np.ndarray p_production, np.ndarray p_consumption,
                            cycle_worker, chunk_size, str sim_id):

    # Initialise the parameters again as C-Variables. Not sure if this is necessary for all variables, but I do not think
    # There is a downside. For the primitive variables the overhead should be minimal. For the arrays it increases
    # performance. For the variables used in the for-loop head it also increases performance.
    cdef int simulation_time = p_simulation_time
    cdef int column_count = p_column_count
    cdef int production_time_resolution = p_production_time_resolution
    cdef int consumption_time_resolution = p_consumption_time_resolution
    cdef np.ndarray[int, ndim = 1] production = p_production
    cdef np.ndarray[int, ndim = 1] consumption = p_consumption
    cdef np.ndarray[int, ndim = 2] int_results = np.zeros((chunk_size, column_count)).astype(int)
    cdef int cycle_count = 0
    simulation_data = SimulationData()

    # Driving Loop for the simulation.
    for cycle_count in range(simulation_time):

        # Check whether the end of a cycle has been reached.
        if cycle_count % chunk_size == 0:
            if cycle_count != 0:
                # Skip first iteration of each cycle, as the cycle only just began.

                # Store and evaluate intermediary results.
                evaluate_and_store_intermediary_results(int_results, sim_id, simulation_data)

                # Reset array for new data of new cycle.
                int_results = np.zeros((chunk_size, p_column_count)).astype(int)

        # Check whether the time delta for the production input has elapsed.
        if cycle_count % production_time_resolution == 0:
            idx = int(cycle_count / production_time_resolution)
            current_production = production[idx]

        # Check whether the time delta for the consumption input has elapsed.
        if cycle_count % consumption_time_resolution == 0:
            idx = int(cycle_count / consumption_time_resolution)
            current_consumption = consumption[idx]

        # Call Java Cycle worker to execute the cycle in java.
        cycle_worker.executeCycle(current_production, current_consumption)

        # Read and store results of the current cycle.
        int_results[cycle_count % chunk_size, 0] = np.datetime64(cycle_worker.getCurrentDateTime(), 's').astype(np.int64)
        int_results[cycle_count % chunk_size, 1] = cycle_worker.getMainActivePower()
        int_results[cycle_count % chunk_size, 2] = cycle_worker.getSupportActivePower()
        int_results[cycle_count % chunk_size, 3] = cycle_worker.getHessActivePower()
        int_results[cycle_count % chunk_size, 4] = cycle_worker.getMainSoc()
        int_results[cycle_count % chunk_size, 5] = cycle_worker.getSupportSoc()
        int_results[cycle_count % chunk_size, 6] = cycle_worker.getHessSoc()
        int_results[cycle_count % chunk_size, 7] = cycle_worker.getGridActivePower()
        int_results[cycle_count % chunk_size, 8] = cycle_worker.getCurrentConsumption()
        int_results[cycle_count % chunk_size, 9] = cycle_worker.getCurrentProduction()
        int_results[cycle_count % chunk_size, 10] = cycle_worker.getMainSoCState()
        int_results[cycle_count % chunk_size, 11] = cycle_worker.getSupportSoCState()

    # Store and evaluate the final remaining results of the current cycle.
    evaluate_and_store_intermediary_results(int_results[:(cycle_count % chunk_size)], sim_id, simulation_data)

def fast_cycle_loop_java(column_count, int simulation_time, int production_time_resolution,
                     int consumption_time_resolution, cycle_worker, np.ndarray production, np.ndarray consumption,
                            chunk_size, str sim_id):
    """
    Wrapper callable from python to call C-Function c_fast_cycle_loop_java.

    c_fast_cycle_loop_java drives the simulation. For each cycle it initiates the necessary execution of the java code.
    After the execution of the java code it reads and stores the result. After a chunk of size chunk_size of the
    simulation has been completed, it will initiate storing the intermediary results on the hard drive.

    :param column_count: Number of column in the result data frame.
    :param simulation_time: Duration of simulation given in [s].
    :param production_time_resolution: Time delta in [s] between to entries of the input consumption data.
    :param consumption_time_resolution: Time delta in [s] between to entries of the input production data.
    :param cycle_worker: Reference to Java Object of Cycle Worker.
    :param production: np.ndarray containing the input data used for the production in the simulation.
    :param consumption: np.ndarray containing the input data used for the consumption in the simulation.
    :param chunk_size: Chunk size in [s] for memory optimization.
                        The total simulation time is divided in chunks of size <chunk_size>.
                        After on chunk has been processed, the intermediary results will be stored on the hard drive
    :param sim_id: Simulation ID to name result files according to simulation and scenario.
    :return:
    """
    c_fast_cycle_loop_java(p_simulation_time=simulation_time, p_column_count=column_count,
                                     p_production_time_resolution=production_time_resolution,
                                     p_consumption_time_resolution=consumption_time_resolution,
                                     cycle_worker=cycle_worker,
                                     p_production=production, p_consumption=consumption,
                                     chunk_size=chunk_size,
                                     sim_id=sim_id)