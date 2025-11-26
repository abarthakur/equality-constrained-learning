
import datetime
import pickle
import os
import argparse
from loguru import logger
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import wandb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.dual_vars import DualVariableContainer
from src.utils import fix_randomness,get_git_root
from src.problems import GenericProblem
from src.solvers import GradientPDSolver

################################################################################
# SETUP PROBLEM
################################################################################

class MultiClassCrossEntropy:
    def __init__(self):
        pass

    def __call__(self, predictions, targets, no_reduce=False) :
        assert len(targets.shape)==1 and len(predictions.shape)==2
        num_samples, num_classes=predictions.shape
        assert targets.shape[0]==num_samples
        per_sample=-torch.log(predictions[torch.arange(0,num_samples), targets])
        if no_reduce : 
            return per_sample
        else :
            return torch.mean(per_sample)
    
def compute_classification_loss(model, x, y, loss_fn):
    assert y.shape==(x.shape[0], 1) and len(x.shape)==2
    return loss_fn(model(x), y.flatten())


def compute_disparity(model, x, y, g, num_groups, filterfunc, sigtemp,equality=True, eps=None):
    assert y.shape[1]==g.shape[1]==1 and x.shape[0]==y.shape[0]==g.shape[0]
    
    y,g = y.flatten(), g.flatten()
    mask = filterfunc(y)
    out = model(x)
    assert out.shape[1]==2
    probs_one = out[mask][:,1]
    ucmetric=torch.mean(F.sigmoid(sigtemp*(probs_one - 0.5)))
    cmetrics = []
    for group in range(num_groups):
        gmask=(g==group) & mask
        probs_one = out[gmask][:,1]
        cmet=torch.mean(F.sigmoid(sigtemp*(probs_one - 0.5)))
        cmetrics.append(cmet)
    constraints = torch.stack(cmetrics)
    
    # transform slacks
    constraints = constraints - ucmetric
    if equality : 
        assert constraints.shape == (num_groups,)
        return constraints
    # else
    assert type(eps)==float
    constraints = torch.concatenate([constraints-eps, -constraints - eps], axis=0)
    assert constraints.shape == (2*num_groups,)
    return constraints


class FairnessProblem(GenericProblem):
    def __init__(self, constraints, equalities, tols, sigtemp, num_groups):
        super().__init__()
        assert len(constraints) == len(equalities) and len(constraints) <=3
        assert len(tols)==0 or len(tols)==len(constraints)
        self.loss_fn = MultiClassCrossEntropy()
        self.sigtemp = sigtemp
        self.num_groups = num_groups
        
        # add classification loss
        self.add_constraint(
            lambda model, x, y: compute_classification_loss(model, x, y, self.loss_fn),
            ctype="objective",
            in_indices=("x", "y"),
            name="mc_cross_entropy"
        )
        
        assert len(constraints) == len(set(constraints))
        
        eqdict, toldict = {}, {}
        
        def dp_mask(y):
            return torch.ones_like(y, dtype=torch.bool, device=y.device)
        
        if "dp" in constraints:
            idx=constraints.index("dp")
            eqdict["dp"] = equalities[idx]
            toldict["dp"] =tols[idx] if not equalities[idx] else None
            self.add_constraint(
                lambda model, x, y, g: compute_disparity(model, x, y, g, self.num_groups, dp_mask, self.sigtemp, equality=eqdict['dp'], eps=toldict['dp']),
                ctype="vector",
                isequality=equalities[idx],
                in_indices=("x", "y","g"),
                name="dp",
                num_constraints=num_groups if equalities[idx] else 2*num_groups
            )
            
        def eo0_mask(y):
            return y==0
        
        if "eo0" in constraints:
            idx=constraints.index("eo0")
            eqdict["eo0"] = equalities[idx]
            toldict["eo0"] =tols[idx] if not equalities[idx] else None
            self.add_constraint(
                lambda model, x, y, g: compute_disparity(model, x, y, g, self.num_groups, eo0_mask, self.sigtemp, equality=eqdict['eo0'], eps=toldict['eo0']),
                ctype="vector",
                isequality=equalities[idx],
                in_indices=("x", "y","g"),
                name="eo0",
                num_constraints=num_groups if equalities[idx] else 2*num_groups
            )
            
        def eo1_mask(y):
            return y==1
        
        if "eo1" in constraints:
            idx=constraints.index("eo1")
            eqdict["eo1"] = equalities[idx]
            toldict["eo1"] =tols[idx] if not equalities[idx] else None
            self.add_constraint(
                lambda model, x, y, g: compute_disparity(model, x, y, g, self.num_groups, eo1_mask, self.sigtemp, equality=eqdict['eo1'], eps=toldict['eo1']),
                ctype="vector",
                isequality=equalities[idx],
                in_indices=("x", "y","g"),
                name="eo1",
                num_constraints=num_groups if equalities[idx] else 2*num_groups
            )
        
def setup_problem(args):
    return FairnessProblem(constraints=args.cspecs, equalities = args.eqs, tols= args.tols, sigtemp = args.sigtemp, num_groups=args.num_groups)
        
################################################################################
# SETUP DATA
################################################################################

class SubgroupDataset(Dataset):
    def __init__(self, train_dataset):
        self.train_dataset  = train_dataset
        self.num_samples = train_dataset['x'].shape[0]
        self.var_names = self.train_dataset.keys()

    def __getitem__(self, idx):
        item = {k: (self.train_dataset[k][idx]) for k in self.var_names}
        return item

    def __len__(self):
        return self.num_samples
    

def load_dataset(args):
    gitroot = get_git_root()
    data_path = os.path.join(gitroot, f"data/fairness/compas-analysis/processed/{args.dataset_file}")
    with open(data_path, "rb") as fi:
        dataset_dict = pickle.load(fi)
    train_dataset=SubgroupDataset(dataset_dict['train'])
    test_dataset=SubgroupDataset(dataset_dict['test'])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=args.pin_memory)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, pin_memory=args.pin_memory)
    return trainloader, testloader
    
################################################################################
# SETUP MODEL AND OPTIM
################################################################################

class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features, predict_vector=True):
        super(LogisticRegression, self).__init__()
        self.num_features=num_features
        self.theta = torch.nn.Parameter(torch.zeros([num_features,1], dtype = torch.float), requires_grad = True)
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype = torch.float) , requires_grad = True)
        self.predict_vector=predict_vector

    def __call__(self, x):
        assert len(x.shape)==2 and x.shape[1]==self.num_features
        prob_1 = self.logit(torch.mm(x, self.theta) + self.bias)
        if self.predict_vector:
            return torch.hstack([1-prob_1, prob_1])
        else:
            return prob_1

    @staticmethod
    def logit(x):
        return 1/(1 + torch.exp(-x))


def setup_model_and_optim(problem, args, DEVICE):
    if DEVICE =="cuda":
        def dpwrapper(mod):
            newmod=torch.nn.DataParallel(mod)
            return newmod
        wrapper = dpwrapper
    else : 
        wrapper = lambda mod : mod
    model=LogisticRegression(num_features=23, predict_vector=True)
    model = wrapper(model).to(DEVICE)
    
    # Setup optimizers
    primal_optim = torch.optim.Adam(
        model.parameters(),
        lr=args.primal_lr,
        weight_decay=args.primal_wd,
        betas=(args.primal_beta1, args.primal_beta2)
    )
    
    # setup duals 
    duals = DualVariableContainer(problem, DEVICE=DEVICE).to(DEVICE)
    dual_params = duals.parameters()
    dual_optim = torch.optim.Adam(
        dual_params,
        lr=args.dual_lr,
        weight_decay=args.dual_wd,
        betas=(args.dual_beta1, args.dual_beta2),
        maximize=True
    ) if len(dual_params) > 0 else None
    
    if args.use_dual_sgd : 
        dual_optim = torch.optim.SGD(
            dual_params,
            lr=args.dual_lr,
            weight_decay=args.dual_wd,
            maximize=True
        ) if len(dual_params) > 0 else None
    
    return model, duals, primal_optim, dual_optim


################################################################################
# SETUP EVALUATORS AND CHECKPOINTS
################################################################################

def log_models(args, run, solver):
    run_id = run.id
    os.makedirs(args.output_dir, exist_ok=True)
    # Save and log primal model
    primal_path = os.path.join(args.output_dir, f"{run_id}_{step_str}_primal.pt")
    torch.save(solver.model.state_dict(), primal_path)
    step_str = solver.epoch_count if not solver.finished else "final"
    run.log_model(primal_path, name=f"{run_id}_{step_str}_primal")


def setup_evaluators(args, run, DEVICE):
    def log_base_metrics(solver) :
        if solver.total_step_count % args.logging_interval == 0 : 
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
            for dv, _, ctype, _, cons_name, isequality in solver.duals.dual_blocks:
                if ctype == "objective":
                    metrics[f"{cons_name}/residual"] = dv.last_value
                elif ctype == "vector":
                    # Log full tensors for vector constraints
                    metrics[f"{cons_name}/lagrangian"] = dv.last_lagrangian
                    res, duals =dv.last_residuals.cpu().detach().numpy(), dv.last_duals.cpu().detach().numpy()
                    assert res.shape == duals.shape == (dv.num_constraints,)
                    if isequality :
                        for i in range(0, len(res)):                
                            metrics[f"{cons_name}/{i}/residual"] = res[i].item()
                            metrics[f"{cons_name}/{i}/dual"] = duals[i].item()
                    else:
                        # also log the dual diff = dual_plus - dual_minus. 
                        # and the true residual = residual_plus - residual_minus /2 
                        # along with per constraint duals
                        n = len(res) // 2
                        for i in range(n):
                            residual_plus = res[i]
                            residual_minus = res[i + n]
                            dual_plus = duals[i]
                            dual_minus = duals[i + n]
                            metrics[f"{cons_name}/{i}/residual_plus"] = residual_plus.item()
                            metrics[f"{cons_name}/{i}/residual_minus"] = residual_minus.item()
                            metrics[f"{cons_name}/{i}/dual_plus"] = dual_plus.item()
                            metrics[f"{cons_name}/{i}/dual_minus"] = dual_minus.item()
                            metrics[f"{cons_name}/{i}/dual_diff"] = (dual_plus - dual_minus).item()
                            metrics[f"{cons_name}/{i}/true_residual"] = ((residual_plus - residual_minus) / 2).item()
                    
                    metrics[f"{cons_name}/residual_norm"] = np.linalg.norm(res)
                    metrics[f"{cons_name}/residual_mean"] = np.mean(res)
                    metrics[f"{cons_name}/dual_norm"] = np.linalg.norm(duals)
                    metrics[f"{cons_name}/dual_mean"] = np.mean(duals)
                else :
                    raise ValueError
                run.log(metrics, step=solver.total_step_count)      
    return [log_base_metrics, log_constraintwise_metrics]


def evaluate(args, run, model, duals, testloader, total_step_count, DEVICE):
    model.eval()
    metrics={}
    with torch.no_grad():
        # get the full dataset, its a small one anyway
        all_x, all_y, all_g = [], [], []
        for batch in testloader:
            x = batch['x'].to(DEVICE)
            y = batch['y'].to(DEVICE)
            g = batch['g'].to(DEVICE)
            all_x.append(x)
            all_y.append(y)
            all_g.append(g)
        x = torch.cat(all_x, dim=0)
        y = torch.cat(all_y, dim=0).flatten()
        g = torch.cat(all_g, dim=0).flatten()
        
        # compute accuracy
        out = model(x)
        assert out.shape[1]==2 and out.shape[0]==x.shape[0]
        pred = torch.argmax(out, dim=1)
        acc = (pred == y).float().mean().item()
        metrics["test/pop/accuracy"] = acc
        
        # Per-group accuracy
        num_groups = int(g.max().item()) + 1
        group_accs = []
        for group in range(num_groups):
            mask = (g == group)
            if mask.sum() > 0:
                group_acc = (pred[mask] == y[mask]).float().mean().item()
            else:
                group_acc = float('nan')
            metrics[f"test/group_{group}/accuracy"] = group_acc
            group_accs.append(group_acc)
        acc_disparity = max(group_accs) - min(group_accs)
        metrics["test/accuracy_disparity"] = acc_disparity

        # Save last duals for each constraint group
        for i, cons in enumerate(args.cspecs):
            eq = args.eqs[i]
            dual_block = None
            for block in duals.dual_blocks:
                if block[4] == cons:
                    dual_block = block[0]
                    break
            if dual_block is not None:
                if eq:
                    for j in range(dual_block.num_constraints):
                        metrics[f"test/{cons}/{j}/dual"] = dual_block.last_duals[j].item()
                else:
                    n = dual_block.num_constraints // 2
                    for j in range(n):
                        dual_plus = dual_block.last_duals[j].item()
                        dual_minus = dual_block.last_duals[j + n].item()
                        metrics[f"test/{cons}/{j}/dual_plus"] = dual_plus
                        metrics[f"test/{cons}/{j}/dual_minus"] = dual_minus
                        metrics[f"test/{cons}/{j}/dual"] = dual_plus - dual_minus

        # Compute exact and approx rates for each constraint type
        sigtemp = args.sigtemp
        def dp_mask(y): return torch.ones_like(y, dtype=torch.bool, device=y.device)
        def eo0_mask(y): return y == 0
        def eo1_mask(y): return y == 1
        mask_funcs = {"dp": dp_mask, "eo0": eo0_mask, "eo1": eo1_mask}
        for cons in ["dp", "eo0", "eo1"]:
            mask_func = mask_funcs[cons]
            mask = mask_func(y)
            approx_rates = []
            exact_rates = []
            for group in range(num_groups):
                gmask = (g == group) & mask
                if gmask.sum() == 0:
                    continue
                probs_one = out[gmask][:, 1]
                approx_rate = torch.mean(F.sigmoid(sigtemp * (probs_one - 0.5))).item()
                exact_rate = torch.mean((probs_one > 0.5).float()).item()
                metrics[f"test/{cons}/group_{group}/approx_rate"] = approx_rate
                metrics[f"test/{cons}/group_{group}/exact_rate"] = exact_rate
                approx_rates.append(approx_rate)
                exact_rates.append(exact_rate)
            
            metrics[f"test/{cons}/approx_rate_disparity"] = max(approx_rates) - min(approx_rates)
            metrics[f"test/{cons}/exact_rate_disparity"] = max(exact_rates) - min(exact_rates)
            # Population-wide
            probs_one = out[mask][:, 1]
            approx_rate = torch.mean(F.sigmoid(sigtemp * (probs_one - 0.5))).item()
            exact_rate = torch.mean((probs_one > 0.5).float()).item()
            metrics[f"test/{cons}/pop/approx_rate"] = approx_rate
            metrics[f"test/{cons}/pop/exact_rate"] = exact_rate
    run.log(metrics, step=total_step_count)
    run.summary['final_evaluation'] = metrics
    

def termination_criteria(args):
    def terminate_on_lag(solver):
        # Keep a buffer of the last 100 lagrangian values
        if not hasattr(solver, "_lag_window"):
            solver._lag_window = []
        lag = solver.duals.last_lagrangian
        solver._lag_window.append(lag)
        if len(solver._lag_window) > args.termwindow:
            solver._lag_window.pop(0)
            
        # Keep a buffer of the last 100 mean values
        if not hasattr(solver, "_lag_mean_buffer"):
            solver._lag_mean_buffer = []
        lag_mean=np.mean(solver._lag_window).item()
        solver._lag_mean_buffer.append(lag_mean)
        # Check if we have enough values
        if len(solver._lag_mean_buffer) < args.termbuffer:
            return False
        elif len(solver._lag_mean_buffer) > args.termbuffer:
            assert len(solver._lag_mean_buffer)==args.termbuffer+1
            solver._lag_mean_buffer.pop(0)
        return np.max(solver._lag_mean_buffer) - np.min(solver._lag_mean_buffer) < args.termtol
    if args.lagterm:
        return [terminate_on_lag]
    else:
        return []

################################################################################
# MAIN TRAINING LOOP
################################################################################

def train(args=None):
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
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        fix_randomness(args.seed, cudnn_deterministic=True) 
        
    matplotlib.use('agg')
    problem = setup_problem(args)
    trainloader, testloader= load_dataset(args)
    model, duals, primal_optim, dual_optim = setup_model_and_optim(problem, args, DEVICE)
    with run as run:
        evaluators = setup_evaluators(args, run, DEVICE)
        tcriteria = termination_criteria(args)
        solver = GradientPDSolver(problem, model, duals, primal_optim, dual_optim, trainloader, evaluators=evaluators, termination_criteria= tcriteria, device=DEVICE, primal_scheduler =None)
        model, duals = solver.train(args.num_epochs)
        evaluate(args, run, model, duals, testloader, solver.total_step_count, DEVICE)
        if args.save_final:
            log_models(args, run, solver)

################################################################################
# CLI ARGS AND MAIN()
################################################################################

def build_parser():
    parser = argparse.ArgumentParser()

    # problem
    parser.add_argument('--cspecs', nargs='*',  choices=["dp", "eo0", "eo1"], default=[], type=str,
                        help="Constraint types to enforce: dp, eo0, eo1 (population-level, equalized odds for y=0/y=1).")
    parser.add_argument('--tols', nargs='*', default=[], type=float,
                        help="Tolerance(s) for each non-equality constraint (floats).")
    parser.add_argument('--eqs', nargs='*',  choices=["true", "false"], default=[], type=str,
                        help="Whether each constraint is equality ('true') or inequality ('false').")
    parser.add_argument('--sigtemp', type=float, default=8.0,
                        help="Sigmoid temperature used to approximate indicator (higher -> steeper).")
    parser.add_argument('--num_groups', type=int, default=4, help="Number of protected groups in the data.")
    
    # primal
    parser.add_argument('--batch_size', type=int, default=139, help="Training batch size.")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--primal_lr', type=float, default=1e-3, help="Primal optimizer learning rate.")
    parser.add_argument('--primal_wd', type=float, default=0.0, help="Primal optimizer weight decay.")
    parser.add_argument('--primal_beta1', type=float, default=0.9, help="Primal Adam beta1.")
    parser.add_argument('--primal_beta2', type=float, default=0.999, help="Primal Adam beta2.")
    
    # duals
    parser.add_argument('--dual_lr', type=float, default=1e-3, help="Dual optimizer learning rate.")
    parser.add_argument('--dual_wd', type=float, default=0.0, help="Dual optimizer weight decay.")
    parser.add_argument('--dual_beta1', type=float, default=0.9, help="Dual Adam beta1.")
    parser.add_argument('--dual_beta2', type=float, default=0.999, help="Dual Adam beta2.")
    parser.add_argument('--use_dual_sgd', action='store_true', default=False,
            help="Use SGD for dual optimization instead of Adam.")
    
    # eval
    parser.add_argument('--dataset_file', type=str, default='', help='Filename under data/fairness/compas-analysis/processed to load (pickle).')
    parser.add_argument('--logging_interval', type=int, default=1, help='Log metrics every N iterations.')
    parser.add_argument('--save_final', action='store_true', default=False, help="Save final model checkpoints.")
    
    # termination
    parser.add_argument('--lagterm', action='store_true', default=False,
            help="Enable termination based on lagrangian stability.")
    parser.add_argument('--termwindow', type=int, default=100, help="Window length (steps) to average lagrangian over.")
    parser.add_argument('--termbuffer', type=int, default=100, help="Number of averaged values to consider for termination.")
    parser.add_argument('--termtol', type=float, default=1e-5, help="Tolerance for lagrangian variation to stop training.")
    
    
    # wandb
    parser.add_argument('--entity', type=str, required=True,
                        help='Wandb entity/account name.')
    parser.add_argument('--project_name', type=str, default='reproduce',
                        help='Wandb project name.')
    parser.add_argument('--wandbgrp', type=str, default='dempar', help='WandB run group.')
    parser.add_argument('--namestr', type=str, default='', help='Optional string to include in the run name.')
    
    # misc
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Computation device: 'cuda' or 'cpu'.")
    parser.add_argument('--pin_memory', action='store_true', default=False,
            help="Pin memory in DataLoader for faster host->GPU transfers.")
    parser.add_argument('--eval_batch_size', type=int, default=4500, help="Batch size used during evaluation.")
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs and checkpoints.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (None for random).')
    return parser
    
if __name__ == "__main__":
    parser = build_parser()
    args= parser.parse_args()
    args.eqs = [s=='true' for s in args.eqs]
    train(args)