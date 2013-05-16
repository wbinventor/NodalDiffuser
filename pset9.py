from solver import *
from materializer import *
from matricizer import *

materializer = Materializer()
matricizer = Matricizer(materializer)
solver = Solver(matricizer)

solver.solveCMFD()
solver.generateCMFDFlux()
solver.plotCMFDFlux()

solver.solveNEM2()
solver.generateNEM2Flux()
solver.plotNEM2Flux()

solver.solveNEM4()
solver.generateNEM4Flux()
solver.plotNEM4Flux()

solver.generateReferenceFlux()
solver.plotReferenceFlux()
solver.plotAllFluxes(1)
solver.plotAllFluxes(2)
