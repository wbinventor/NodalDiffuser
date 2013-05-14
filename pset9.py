from solver import *
from materializer import *
from matricizer import *

materializer = Materializer()
matricizer = Matricizer(materializer)
solver = Solver(matricizer)

matricizer.initializeCMFDDestructionMatrix()
matricizer.initializeCMFDProductionMatrix()
matricizer.initializeNEM4thOrderCoeffMatrix(keff=1.0)
matricizer.initializeNEM2ndOrderCoeffMatrix()

#matricizer.spyCMFDProductionMatrix()
#matricizer.spyCMFDDestructionMatrix()
#matricizer.spyNEM4thOrderCoeffMatrix()
#matricizer.spyNEM2ndOrderCoeffMatrix()

#solver.solveCMFD()
#solver.plotCMFDFlux()

solver.solveNEM4()
