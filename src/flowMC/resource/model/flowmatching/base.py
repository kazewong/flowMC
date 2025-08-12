import equinox as eqx
from flowMC.resource.base import Resource

class Solver:
    pass
    
class Path:
    pass
    
class FlowMatchingModel(eqx.Module, Resource):
    
    def __init__(self, solver: Solver, path: Path):
        self.solver = solver
        self.path = path

