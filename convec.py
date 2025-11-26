import argparse
import datetime
import glob
import os
import torch
import torch.nn as nn
import wandb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from loguru import logger
from torch.optim.lr_scheduler import StepLR
from src.solvers import GradientPDSolver
from src.dual_vars import DualVariableContainer
from src.models.simple import ScalarMLP  
from src.utils import fix_randomness, get_git_root, get_histogram
from src.problems import GenericProblem

################################################################################
# SETUP PROBLEM
################################################################################

class GenericPDEProblem(GenericProblem):
    def __init__(self, domain):
        """
        domain: dict with keys like 'x', 't' each mapping to (low, high)
        """
        super().__init__()
        self.domain = domain


def convection_pde_residual(model, x, t, beta):
    """
    Compute the PDE residual for the 1D convection equation:
        ∂u/∂t + beta* ∂u/∂x = 0

    Returns:
    - residual: Tensor representing the PDE residual at each (x, t)
    """

    # Ensure x and t require gradients for autograd
    x.requires_grad_(True)
    t.requires_grad_(True)

    # Stack x and t and pass through model
    inputs = torch.concatenate([x, t], dim=-1)
    u = model(inputs)

    # Compute ∂u/∂x
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    # Compute ∂u/∂t
    du_dt = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    
    assert du_dx.shape == du_dt.shape == x.shape == t.shape

    # PDE residual: ∂u/∂t + c ∂u/∂x
    residual = du_dt + beta * du_dx

    return residual


def convection_bc_residual(model, t, x_left, x_right):
    """
    Compute the residual for the periodic boundary condition:
        u(x_left, t) = u(x_right, t)

    Returns:
    - residual: Tensor of differences u(x_left, t) - u(x_right, t)
    """

    # Expand to match shape of t
    x_l = torch.full_like(t, x_left)
    x_r = torch.full_like(t, x_right)

    inputs_l = torch.concatenate([x_l, t], dim=-1)
    inputs_r = torch.concatenate([x_r, t], dim=-1)

    u_l = model(inputs_l)
    u_r = model(inputs_r)

    residual = u_l - u_r
    assert residual.shape[1] == 1 and residual.shape[0] == t.shape[0]
    return residual


def convection_ic_residual(model, x, h_fn=None):
    """
    Compute the residual for the initial condition:
        u(x, 0) = h(x)

    Returns:
    - residual: Tensor of differences u(x, 0) - h(x)
    """
    if h_fn is None:
        h_fn = torch.sin

    t0 = torch.zeros_like(x)
    inputs = torch.concatenate([x, t0], dim=-1)
    u = model(inputs)

    target = h_fn(x)
    residual = u - target
    return residual


class ConvectionProblem(GenericPDEProblem):
    
    def __init__(self, beta=1.0, ic_case ='sin', x_bounds=[0.0, 2*torch.pi], t_right=1.0, lambdas =[1.0, 1.0, 1.0], ctypes = ["vec", "vec", "vec"], device='cpu'):
        super().__init__(domain={"x": (x_bounds[0], x_bounds[1]), "t": (0.0, t_right)})
        self.constraint_names = ['ic', 'bc', 'pde']
        self.beta = beta
        self.ic_case = ic_case
        if self.ic_case == 'sin':
            self.init_fn = torch.sin
        self.x_left = x_bounds[0]
        self.x_right = x_bounds[1]
        self.input_dim = 2
        self.input_indices = ("x","t",)
        self.lambdas = lambdas 
        self.ctypes = ctypes
        
        # configure ic constraint 
        def ic_obj(model, x):
            res = convection_ic_residual(model, x, h_fn=self.init_fn)
            obj = self.lambdas[0]*torch.mean(res**2)
            return obj
        
        if self.ctypes[0] == 'obj' :
            self.add_constraint(
                lambda model, x : ic_obj(model, x),
                ctype="objective",
                in_indices=("x",),
                name="ic"
            )
        elif self.ctypes[0] == 'vec' :
            self.add_constraint(
                lambda model, x : ic_obj(model, x).reshape(1,),
                ctype="vector",
                in_indices=("x",),
                name="ic",
                isequality = True,
                num_constraints=1
            )
        else : 
            raise ValueError
            
        # configure bc constraint 
        def bc_obj(model, t):
            res = convection_bc_residual(model, t, x_left=self.x_left, x_right=self.x_right)
            obj = self.lambdas[1]*torch.mean(res**2)
            return obj
        
        if self.ctypes[1] == 'obj' :
            self.add_constraint(
                lambda model, t : bc_obj(model, t),
                ctype="objective",
                in_indices=("t",),
                name="bc"
            )
        elif self.ctypes[1] == 'vec' :
            self.add_constraint(
                lambda model, t : bc_obj(model, t).reshape(1,),
                ctype="vector",
                in_indices=("t",),
                name="bc",
                isequality = True,
                num_constraints=1
            )
        else : 
            raise ValueError
        
        
        # configure pde cons
        def pde_obj(model, x,  t):
            res = convection_pde_residual(model, x, t, beta=self.beta)
            obj = self.lambdas[2]*torch.mean(res**2)
            return obj
        
        if self.ctypes[2] == 'obj' :
            self.add_constraint(
                lambda model, x, t : pde_obj(model, x, t),
                ctype="objective",
                in_indices=("x","t",),
                name="pde"
            )
        elif self.ctypes[2] == 'vec' :
            self.add_constraint(
                lambda model, x, t : pde_obj(model, x, t).reshape(1,),
                ctype="vector",
                in_indices=("x","t",),
                name="pde",
                isequality = True,
                num_constraints=1
            )
        else : 
            raise ValueError
        
        if 'obj' not in self.ctypes :
            self.add_constraint(
                lambda model: torch.tensor(0.0, dtype=torch.float32, requires_grad=False, device = device),
                ctype="objective",
                in_indices=(),
                name="nullobj"
            )
        

################################################################################
# SETUP DATA
################################################################################


class PDEDataset:
    def __init__(self, problem, num_samples,  device = 'cpu'):
        """
        problem: instance of GenericPDEProblem
        num_samples: number of samples in the dataset
        mode: 'uniform' (resample each item) or 'fixed' (precompute dataset)
        """
        assert isinstance(problem, GenericPDEProblem)

        self.num_samples = num_samples

        # Get variable names and bounds from problem.domain
        self.var_names = list(problem.domain.keys())
        self.bounds = [problem.domain[name] for name in self.var_names]
        self.device = device


    def _sample(self, count):
        # Prepare a dict for sampled variables
        samples = {}

        # Collocation: sample all variables uniformly in their bounds
        for name, (low, high) in zip(self.var_names, self.bounds):
            samples[name] = torch.rand(count, device=self.device) * (high - low) + low
        return samples

def make_dummy_loader(dataset, bsize):
    class RandomBatch:
        def __init__(self):
            self.done=False
            pass

        def __iter__(self):
            return self  # An iterator returns itself

        def __next__(self):
            if self.done :
                self.done=False
                raise StopIteration
            else: 
                sample= dataset._sample(bsize)
                self.done=True
                return {k: v.unsqueeze(-1) for k, v in sample.items()}
    return RandomBatch()


def setup_test_set(args, DEVICE):
    """
    Searches for a .npy file in gitroot/data/<problem_name> matching the format:
    solution_{r}_{y0}_*.npy for the 'growth' problem.
    If multiple matches, picks the one with the largest number of grid points.
    Loads the npy file (should be a dict with keys 'solution' and 'indices'), converts all arrays to torch tensors.
    Returns a dict with 'solution' (tensor) and 'grid' (dict of tensors).
    """
    gitroot = get_git_root()
    data_dir = os.path.join(gitroot, "data/pdelearn", 'convection')

    testdata = None
    search_path = os.path.join(data_dir, args.testfile)
    files = glob.glob(search_path)
    assert files is not None and len(files)==1
    chosen_file = files[-1]
    print(f"Using test set: {chosen_file}")
    testdata = np.load(chosen_file, allow_pickle=True).item()
    

    # Convert arrays to torch tensors
    def convert(data):
        if data is None : return None
        solution = torch.from_numpy(data["solution"].astype('float32')).to(DEVICE)
        grid = {k: torch.from_numpy(v.astype('float32')).to(DEVICE) for k, v in data["grid"].items()}
        return {"solution": solution, "grid": grid}

    return convert(testdata)


################################################################################
# SETUP MODEL AND OPTIM
################################################################################

def setup_model_and_optim(problem, args, DEVICE):
    if DEVICE =="cuda":
        def dpwrapper(mod):
            newmod=torch.nn.DataParallel(mod)
            # patch attributes that are necessary 
            newmod.num_features = mod.num_features
            newmod.input_indices = mod.input_indices
            newmod.device = DEVICE
            return newmod
        wrapper = dpwrapper
    else : 
        wrapper = lambda mod : mod
    # Setup primal model
    model = wrapper(ScalarMLP(
        num_features=problem.input_dim,
        input_indices=problem.input_indices,
        activation=nn.Tanh(),
        num_hidden_layers=args.primal_layers,
        layer_widths=[args.primal_width] * args.primal_layers, 
        device=DEVICE # the hacks continue (for clone, freeze model)
    ))    
    model = model.to(DEVICE)
    print("Primal models")
    print(model)

    # Setup dual variable container
    duals = DualVariableContainer(problem, DEVICE=DEVICE).to(DEVICE)
    # Setup optimizers
    primal_optim = torch.optim.Adam(
        model.parameters(),
        lr=args.primal_lr,
        weight_decay=args.primal_wd,
        betas=(args.primal_beta1, args.primal_beta2)
    )
    primal_scheduler=None
    if args.schedule_lr : 
        primal_scheduler = StepLR(primal_optim, step_size=args.primal_lr_step, gamma=args.primal_lr_decay)
   
    dual_params = duals.parameters()
    base_dual_optim = torch.optim.Adam(
        dual_params,
        lr=args.dual_lr,
        weight_decay=args.dual_wd,
        betas=(args.dual_beta1, args.dual_beta2),
        maximize=True
    ) if len(dual_params) > 0 else None
    
    dual_model_optim=None

    return model, duals, primal_optim, base_dual_optim, dual_model_optim, primal_scheduler

################################################################################
# SETUP EVALUATORS AND CHECKPOINTS
################################################################################

def log_models(args, run, solver):
    run_name = run.name
    run_id = run.id
    os.makedirs(args.output_dir, exist_ok=True)
    # Save and log primal model
    primal_path = os.path.join(args.output_dir, f"{run_name}_{run_id}_primal.pt")
    torch.save(solver.model.state_dict(), primal_path)
    step_str = solver.epoch_count if not solver.finished else "final"
    run.log_model(primal_path, name=f"{run_name}_{run_id}_{step_str}_primal")

def get_diagnostic_plot_dim2(step_count, inputs, solution_array):
    matplotlib.use('agg')
    x_key, y_key = list(inputs.keys())
    x = inputs[x_key].cpu().detach().numpy()
    y = inputs[y_key].cpu().detach().numpy()
    assert x.shape[1]==1 and y.shape[1]==1 and len(solution_array.shape)==1
    assert type(solution_array)==np.ndarray
    if not (np.allclose(np.diff(x), np.diff(x)[0]) and np.allclose(np.diff(y), np.diff(y)[0])):
        print("Warning: x or y are not equally spaced.")
        raise Exception()
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx, ny = len(x_unique), len(y_unique)
    
    # Assume solution_array is shape (len(y), len(x)) or flatten and reshape
    solution_array = solution_array.reshape(ny, nx)
    fig, ax = plt.subplots()
    extent = [x.min(), x.max(), y.min(), y.max()]
    im = ax.imshow(solution_array, extent=extent, origin='lower', aspect='auto')
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"Step {step_count}")
    fig.colorbar(im, ax=ax, label="Solution")
    out = wandb.Image(fig)
    plt.close(fig)
    return out


def setup_evaluators(args, run, test_dataset, DEVICE):

    def log_base_metrics(solver) :
        if solver.total_step_count % args.logging_interval == 0 : 
            # todo : gradient norms
            lag=solver.duals.last_lagrangian
            obj = solver.duals.last_objective
            primal_parameter_norm = torch.norm(torch.cat([p.view(-1) for p in solver.model.parameters()]), 2).item()
            dual_params =solver.duals.parameters() 
            dual_norm = torch.norm(torch.cat([p.view(-1) for p in dual_params]), 2).item() if len(dual_params)> 0 else 0.0 
            run.log({
                "base/lagrangian" : lag,
                "base/objective" : obj, 
                "base/lag_minus_loss" : lag - obj, 
                "base/primal_parameter_norm" : primal_parameter_norm, 
                "base/dual_base_norm" : dual_norm, 
                "base/full_minmax_objective" : lag + args.primal_wd * primal_parameter_norm - args.dual_wd * dual_norm,
                "base/primal_lr" : args.primal_lr if solver.primal_scheduler is None else solver.primal_scheduler.get_last_lr()[0]
            }, step = solver.total_step_count)
            logger.info(f"Step {solver.total_step_count} | Epoch {solver.epoch_count} | Lagrangian: {lag:.6f}")
        return None
    
    def log_constraintwise_metrics(solver):
        if solver.total_step_count % args.logging_interval == 0:
            metrics = {}
            for dv, _, ctype, _, cons_name, _ in solver.duals.dual_blocks:
                if ctype == "objective":
                    metrics[f"{cons_name}/residual"] = dv.last_value
                elif ctype == "vector":
                    # Log full tensors for vector constraints
                    metrics[f"{cons_name}/lagrangian"] = dv.last_lagrangian
                    res, duals =dv.last_residuals.cpu().detach().numpy(), dv.last_duals.cpu().detach().numpy()
                    assert res.shape == duals.shape == (dv.num_constraints,)
                    for i in range(0, len(res)):                
                        metrics[f"{cons_name}/{i}/residual"] = res[i].item()
                        metrics[f"{cons_name}/{i}/dual"] = duals[i].item()
                    metrics[f"{cons_name}/residual_norm"] = np.linalg.norm(res)
                    metrics[f"{cons_name}/residual_mean"] = np.mean(res)
                    metrics[f"{cons_name}/dual_norm"] = np.linalg.norm(duals)
                    metrics[f"{cons_name}/dual_mean"] = np.mean(duals)
                else :
                    raise ValueError
                run.log(metrics, step=solver.total_step_count)
    
    def checkpoint_models(solver):
        if not solver.finished and (solver.epoch_count % args.checkpoint_interval != 0 or solver.epoch_step_count != 0):
            return
        log_models(args, run, solver)
       
    inputs = test_dataset['grid']
    solution = test_dataset['solution']
        
    # Collect model inputs using model.input_indices, which should correspond to keys for the dictionary inputs
    # Extract the input features in the correct order
    input_keys = ["x","t"]
    model_inputs = torch.concatenate([inputs[key] for key in input_keys], dim=1)
    run.log({f"test/ground_truth_img" : get_diagnostic_plot_dim2(0, inputs, solution.detach().cpu().numpy().flatten())}, step=0)
                    
    def evaluate_model(solver):
        # Only evaluate at the specified interval and at the start of an epoch
        continue_condition = solver.finished or (solver.epoch_count % args.evaluation_interval == 0 and solver.epoch_step_count == 0) or (solver.epoch_count % args.plotting_interval == 0 and solver.epoch_step_count == 0)
        if not continue_condition:
            return

        assert model_inputs.ndim ==2 and solver.model.num_features == model_inputs.shape[-1]
        # Use batching to avoid memory bottleneck
        batch_size = args.eval_batch_size
        model_solutions = []
        with torch.no_grad():
            solver.model.eval()
            for i in range(0, model_inputs.shape[0], batch_size):
                batch_inputs = model_inputs[i:i+batch_size]
                batch_solution = solver.model(batch_inputs)
                model_solutions.append(batch_solution)
        model_solution = torch.cat(model_solutions, dim=0)

        assert model_solution.shape == solution.shape

        # Compute L2 error and relative L2 error
        l2_error = torch.sqrt(torch.sum((model_solution - solution)**2)).item()
        rel_l2_error = l2_error / torch.sqrt(torch.sum(solution**2)).item()
        
        # Log metrics and model solution
        run.log({
            "test/l2_error": l2_error,
            "test/rel_l2_error": rel_l2_error,
        }, step=solver.total_step_count)
        
        if solver.epoch_count % args.plotting_interval == 0 : 
            run.log({f"test/model_solution_hist" : get_histogram(solver.total_step_count, model_solution.detach().cpu().numpy().flatten())}, step=solver.total_step_count)
            run.log({f"test/model_solution_img" : get_diagnostic_plot_dim2(solver.total_step_count, inputs, model_solution.detach().cpu().numpy().flatten())}, step=solver.total_step_count)
        
    return [log_base_metrics, log_constraintwise_metrics, checkpoint_models, evaluate_model]


def setup_termination_criteria(args, DEVICE):
    return []


################################################################################
# MAIN TRAINING LOOP
################################################################################            
def train(args=None):
    # initialise wandb here 
    
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if type(args)==argparse.Namespace : 
        # command line invocation    
        run = wandb.init(
            entity=args.entity,
            project=args.project_name,
            dir=args.output_dir, 
            group=args.wandbgrp,
            name = f"{date_str}_{args.namestr}",
            config=args
        )
        DEVICE = args.device
        args.seed= fix_randomness(args.seed, cudnn_deterministic=True) 
    elif args is None :
        run = wandb.init()
        args= run.config
        assert args.seed is not None
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        fix_randomness(args.seed, cudnn_deterministic=True) 
        
    problem = ConvectionProblem(beta = args.Co_beta, ic_case ='sin', lambdas=args.lambdas, ctypes = args.ctypes, device=DEVICE)
    dataset = PDEDataset(problem, num_samples=args.batch_size, device=DEVICE)
    trainloader = make_dummy_loader(dataset, args.batch_size)
    test_dataset = setup_test_set(args, DEVICE)
    model, duals, primal_optim, base_dual_optim, dual_model_optim, primal_scheduler = setup_model_and_optim(problem, args, DEVICE)
    with run as run:
        evaluators = setup_evaluators(args, run, test_dataset, DEVICE)
        tcriteria = setup_termination_criteria(args, DEVICE)        
        solver = GradientPDSolver(problem, model, duals, primal_optim, base_dual_optim, trainloader, evaluators=evaluators, termination_criteria= tcriteria, device=DEVICE, primal_scheduler =primal_scheduler)
        
        _ = solver.train(args.num_epochs)
        if args.save_final:
            log_models(args, run, solver)
            

################################################################################
# CLI ARGS AND MAIN()
################################################################################

def build_parser():
    parser = argparse.ArgumentParser()
    
    # problem
    parser.add_argument('--Co_beta', type=float, default=1.0, help="Convection beta.")
    parser.add_argument('--ctypes', type=str, choices=['obj', 'vec', 'pw'], nargs=3, required=True,
                        help="Constraint types for (ic, bc, pde); each is 'obj', 'vec' or 'pw'.")
    parser.add_argument('--lambdas', type=float, nargs=3, default=[1.0,1.0,1.0],
                        help="Weights for (ic, bc, pde) loss terms (three floats).")
    
    # primal
    parser.add_argument('--batch_size', type=int, default=139, help="Collocation samples per training batch.")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--primal_layers', type=int, default=2, help="Number of hidden layers in the primal MLP.")
    parser.add_argument('--primal_width', type=int, default=32, help="Hidden layer width in the primal MLP.")
    parser.add_argument('--primal_lr', type=float, default=1e-3, help="Primal Adam learning rate.")
    parser.add_argument('--primal_wd', type=float, default=0.0, help="Primal weight decay (L2).")
    parser.add_argument('--primal_beta1', type=float, default=0.9, help="Adam beta1 for primal optimizer.")
    parser.add_argument('--primal_beta2', type=float, default=0.999, help="Adam beta2 for primal optimizer.")
    parser.add_argument('--schedule_lr', action='store_true', default=False, help="Use LR scheduler for primal optimizer.")
    parser.add_argument('--primal_lr_step', type=int, default=5000, help="Scheduler step size (iterations).")
    parser.add_argument('--primal_lr_decay', type=float, default=0.9, help="LR decay factor (gamma) for scheduler.")

    # dual
    parser.add_argument('--dual_lr', type=float, default=1e-3, help="Dual base Adam learning rate.")
    parser.add_argument('--dual_wd', type=float, default=0.0, help="Dual base weight decay (L2).")
    parser.add_argument('--dual_beta1', type=float, default=0.9, help="Adam beta1 for dual base optimizer.")
    parser.add_argument('--dual_beta2', type=float, default=0.999, help="Adam beta2 for dual base optimizer.")
    
    # Eval and logging
    parser.add_argument('--testfile', type=str, default='', help='Filename or glob for test .npy under data/pdelearn/convection.')
    parser.add_argument('--logging_interval', type=int, default=1, help='Log metrics every N training iterations.')
    parser.add_argument('--evaluation_interval', type=int, default=2, help='Run full evaluation every N epochs.')
    parser.add_argument('--plotting_interval', type=int, default=2, help='Log plots every N epochs.')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Checkpoint every N epochs.')
    parser.add_argument('--eval_batch_size', type=int, default=5000, help='Batch size used during evaluation.')
    parser.add_argument('--save_final', action='store_true', default=False, help="Save final model checkpoints after training.")

    # wandb
    parser.add_argument('--entity', type=str, required=True,
                        help='Wandb entity/account name.')
    parser.add_argument('--project_name', type=str, default='reproduce',
                        help='Wandb project name.')
    parser.add_argument('--wandbgrp', type=str, default='convec', help='WandB run group.')
    parser.add_argument('--namestr', type=str, default='', help='Optional string appended to run name for identification.')
    
    # misc
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to use: 'cuda' or 'cpu'.")
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save checkpoints and outputs.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed. If None, a random seed is chosen.')
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args= parser.parse_args()
    train(args)
