from const import grid_str, consumption_str, hess_active_power_str, production_str


def autarky_hess(row):
    grid = row[grid_str]
    consumption = row[consumption_str]
    hess = row[hess_active_power_str]
    if hess > 0:
        hess = 0
    else:
        hess = -hess
    if grid < 0:
        grid = 0
    if consumption <= 0 and hess <= 0:
        return 100
    result = (1 - (grid / (consumption + hess))) * 100
    result = max(0, min(100, result))
    return result


def self_consumption(row):
    grid = row[grid_str]
    production = row[production_str]
    grid = -grid
    if grid < 0:
        grid = 0
    if production == 0:
        return 0
    result = (1 - (grid / production)) * 100
    result = max(0, min(100, result))
    return result


def mask_only_hess_sell_to_grid(row):
    sell_to_grid = row[1]
    pv_production = row[2]
    hess_discharge = row[3]

    # Only the HESS sells to grid, if energy is sold to grid and there is no PV production.
    return sell_to_grid != 0 and pv_production == 0 and hess_discharge != 0


def get_hess_sells_surplus(row):
    sell_to_grid = row[1]
    pv_production = row[2]
    consumption = row[4]

    if sell_to_grid != 0 and sell_to_grid > pv_production - consumption:

        # All power produced by PV used to cover load => power sold to grid has to be from hess
        if pv_production < consumption:
            return abs(sell_to_grid)
        else:
            # PV covers all consumption & hess discharges => subtract pv power from sold to grid for hess power sold
            return abs(sell_to_grid + (pv_production - consumption))
    else:
        return 0