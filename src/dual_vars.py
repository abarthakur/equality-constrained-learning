from abc import ABC, abstractmethod
import torch

from .problems import GenericProblem

class BaseDualVariable(ABC):
    def __init__(self, in_indices):
        self.in_indices = in_indices  

    @abstractmethod
    def eval(self, batch: dict):
        pass

    @abstractmethod
    def compute_partial_lagrangian(self, model, constraint_fn, batch: dict):
        pass

    def parameters(self):
        return []

    def to(self, device):
        return self


class ObjectiveDualVariable(BaseDualVariable):
    def __init__(self, in_indices: tuple):
        super().__init__(in_indices=in_indices)
        self.last_value = 0.0

    def eval(self, batch):
        # Always return scalar 1.0, broadcast to match residual shape
        return 1.0

    def compute_partial_lagrangian(self, model, constraint_fn, batch):
        args = [batch[k] for k in self.in_indices]
        obj = constraint_fn(model, *args)
        self.last_value = obj.item()
        return obj  # No scaling needed

    def parameters(self):
        return []  # No parameters

    def to(self, device):
        return self  # Nothing to move


class VectorDualVariable(BaseDualVariable):
    
    def __init__(self, num_constraints: int, in_indices: tuple = (), DEVICE='cpu'):
        """
        Dual variable for vector-valued (or scalar) constraints.
        - num_constraints: length of the vector constraint
        - in_indices: names of data inputs needed by the constraint
        """
        super().__init__(in_indices=in_indices)
        self.num_constraints = num_constraints
        self.values = torch.nn.Parameter(torch.zeros(num_constraints))
        
        self.last_residuals = torch.tensor([0.0]*self.num_constraints, device=DEVICE)
        self.last_duals = torch.tensor([0.0]*self.num_constraints, device=DEVICE)
        self.last_lagrangian = 0.0

    def eval(self):
        return self.values  # shape: [K]

    def compute_partial_lagrangian(self, model, constraint_fn, batch: dict):
        """
        residual: shape [K]
        dual_vals: shape [K]
        """
        args = [batch[k] for k in self.in_indices] if self.in_indices else []
        residual = constraint_fn(model, *args)       # shape: [K]
        dual_vals = self.eval()                 # shape: [K]
        assert residual.shape == dual_vals.shape == (self.num_constraints,)
        lag= torch.sum(dual_vals * residual)
        with torch.no_grad():
            self.last_residuals = residual.cpu().detach()
            self.last_duals = dual_vals.cpu().detach()
            self.last_lagrangian = lag.item()
        return lag

    def parameters(self):
        return [self.values]

    def to(self, device):
        if self.values.device != device:
            self.values = torch.nn.Parameter(self.values.data.to(device))
        return self


class DualVariableContainer:
    def __init__(
        self,
        problem: GenericProblem,
        DEVICE = 'cpu'
    ):
        self.dual_blocks = []
        self.problem = problem 
        self.device = DEVICE

        for fn, ctype, in_indices, num_constraints, name, isequality in self.problem.constraints:
            if ctype=="objective":
                dv = ObjectiveDualVariable(in_indices=in_indices)
            elif ctype == "vector":
                dv = VectorDualVariable(num_constraints=num_constraints, in_indices=in_indices, DEVICE=DEVICE)
            else:
                raise ValueError(f"Unsupported constraint type: {ctype}")

            self.dual_blocks.append((dv, fn, ctype, in_indices, name, isequality))
        
        self.last_objective=0.0
        self.last_lagrangian=0.0

    def compute_lagrangian(self, model, batch: dict):
        total = 0.0
        self.last_objective=0.0
        for dv, fn, ctype, *_ in self.dual_blocks:
            part_lagrangian=dv.compute_partial_lagrangian(model, fn, batch)
            total += part_lagrangian
            # tracking
            if ctype == "objective":
                self.last_objective += part_lagrangian.item()
        self.last_lagrangian = total.item() 
        return total
    
    def parameters(self):
        params =[]
        for dv, *_ in self.dual_blocks:
            params.extend(dv.parameters())
        return params

    def to(self, device):
        for dv, *_ in self.dual_blocks:
            dv.to(device)
        return self

