from const import support_capacity, main_capacity
from java_classes import Config

main_config = Config('ess0',  # id
                    'main',  # name
                     True,  # OnGrid
                     main_capacity,  # Capacity
                     1_200,  # Ramp Rate
                     10_000,  # Response Time
                     10_000,  # Inactivity Time
                     10,  # Min. SoC
                     90,  # Max. SoC
                     50,  # Initial SoC
                     90_000,  # Discharge Power
                     78_000  # Charge Power
                     )

support_config = Config('ess1',  # id
                       'support',  # name
                        True,  # OnGrid
                        support_capacity,  # Capacity
                        100_000,  # Ramp Rate
                        2_000,  # Response Time
                        10_000,  # Inactivity Time
                        5,  # Min. SoC
                        95,  # Max. SoC
                        50,  # Initial SoC
                        276_000,  # Discharge Power
                        276_000  # Charge Power
                        )


