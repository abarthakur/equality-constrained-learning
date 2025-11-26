import torch

class GradientPDSolver:
    def __init__(
        self,
        problem,
        model,
        duals,
        model_optimizer,
        dual_optimizer,
        train_loader, 
        evaluators=None,  # list of evaluator_fn
        termination_criteria=None,  # list of terminator_fn
        device="cpu", 
        primal_scheduler=None
    ):
        self.problem = problem
        self.model = model
        self.duals = duals
        self.model_optimizer = model_optimizer
        self.dual_optimizer = dual_optimizer
        self.primal_scheduler = primal_scheduler
        self.train_loader = train_loader
        self.device = torch.device(device)
        self.evaluators = evaluators or []
        self.termination_criteria = termination_criteria or []
        self.total_step_count = 0
        self.epoch_step_count = 0
        self.epoch_count=0
        self.termination_reason = None
        self.finished=False

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.epoch_step_count=0
            for batch in self.train_loader:
                self._evaluate()
                self._step_model(batch)
                self._step_duals(batch)
                self.total_step_count+=1
                self.epoch_step_count+=1
            self.epoch_count+=1
            if self.primal_scheduler is not None :
                self.primal_scheduler.step()
            if self._should_terminate():
                print(f"[Terminated] {self.termination_reason}")
                break
        self.finished=True
        self._evaluate()
        return self.model, self.duals


    def _step_model(self, batch):
        self.model.train()
        self.model_optimizer.zero_grad()
        if self.dual_optimizer is not None : 
            self.dual_optimizer.zero_grad()
        batch = self._to_device(batch)
        lagrangian = self.duals.compute_lagrangian(self.model, batch)
        lagrangian.backward(retain_graph=True)
        self.model_optimizer.step()
        if self.dual_optimizer is not None : 
            self.dual_optimizer.step()
        
        # handle inequalities
        for block in self.duals.dual_blocks:
            dv, ctype, isequality=block[0], block[2], block[-1]
            if ctype != 'vector' or isequality :
                continue
            # project dual vars for inequalities
            dv.values.requires_grad=False
            new_values = torch.maximum(dv.values, torch.tensor(0.0, dtype=torch.float32, requires_grad=False, device=batch['x'].device))
            dv.values.copy_(new_values)
            dv.values.requires_grad=True
        
    def _step_duals(self, batch):
        pass

    def _evaluate(self):
        for eval_fn in self.evaluators:
            eval_fn(self)

    def _should_terminate(self) -> bool:
        for condition in self.termination_criteria:
            if condition(self):
                self.termination_reason = condition.__name__
                return True
        return False
    
    def _to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

