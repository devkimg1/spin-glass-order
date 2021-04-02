from ising_model import Ising2D
from RFMC_stacking import RFMC

# 1024x1024 at T = 1.0
glass = Ising2D(2**10,2**10,1.0)
model = RFMC(glass)

# Run RFMC 1e6 steps
model.run(1e6)
