import inspect

class GenericProblem:
    def __init__(self):
        self.constraints = []

    def add_constraint(self, func, ctype, in_indices, num_constraints=None, name="", isequality=True):
        assert callable(func), "Constraint function must be callable."
        assert isinstance(in_indices, tuple) and all(isinstance(s, str) for s in in_indices), \
            "in_indices must be a tuple of strings."
        assert ctype in ("objective", "vector"), f"Unknown ctype: {ctype}"

        if ctype == "vector":
            assert num_constraints is not None and isinstance(num_constraints, int), \
                "vector constraint requires num_constraints (int)"
        elif ctype == "objective":
            assert num_constraints is None

        sig = inspect.signature(func)
        expected_args = len(in_indices) + 1  # +1 for model
        if len(sig.parameters) != expected_args:
            raise ValueError(f"Constraint function expects {expected_args} arguments (including model), got {len(sig.parameters)}")
        
        self.constraints.append((func, ctype, in_indices, num_constraints, name, isequality))
