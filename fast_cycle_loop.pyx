
import numpy as np
cimport numpy as np

from evaluation import evaluate_and_store_on_the_run
from simulation_data import SimulationData


cdef c_fast_cycle_loop_java(int p_simulation_time, int p_column_count,
                            int p_production_time_resolution, int p_consumption_time_resolution,
                            np.ndarray p_production, np.ndarray p_consumption,
                            cycle_worker, chunk_size, str sim_id):
    cdef int simulation_time = p_simulation_time
    cdef int column_count = p_column_count
    cdef int production_time_resolution = p_production_time_resolution
    cdef int consumption_time_resolution = p_consumption_time_resolution
    cdef np.ndarray[int, ndim = 1] production = p_production
    cdef np.ndarray[int, ndim = 1] consumption = p_consumption
    cdef np.ndarray[int, ndim = 2] int_results = np.zeros((chunk_size, p_column_count)).astype(int)
    cdef int cycle_count = 0
    simulation_data = SimulationData()
    for cycle_count in range(simulation_time):
        if cycle_count % chunk_size == 0:
            if cycle_count != 0:
                evaluate_and_store_on_the_run(int_results, sim_id, simulation_data)
                int_results = np.zeros((chunk_size, p_column_count)).astype(int)
        if cycle_count % production_time_resolution == 0:
            idx = int(cycle_count / production_time_resolution)
            current_production = production[idx]
        if cycle_count % consumption_time_resolution == 0:
            idx = int(cycle_count / consumption_time_resolution)
            current_consumption = consumption[idx]
        cycle_worker.executeCycle(current_production, current_consumption)
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
    evaluate_and_store_on_the_run(int_results[:(cycle_count % chunk_size)], sim_id, simulation_data)

def fast_cycle_loop_java(columns, int simulation_time, int production_time_resolution,
                     int consumption_time_resolution, cycle_worker, np.ndarray production, np.ndarray consumption,
                            chunk_size, str sim_id):
    c_fast_cycle_loop_java(p_simulation_time=simulation_time, p_column_count=len(columns),
                                     p_production_time_resolution=production_time_resolution,
                                     p_consumption_time_resolution=consumption_time_resolution,
                                     cycle_worker=cycle_worker,
                                     p_production=production, p_consumption=consumption,
                                     chunk_size=chunk_size,
                                     sim_id=sim_id)