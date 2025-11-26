
import datetime
import subprocess
import sys
import os
import argparse
from loguru import logger
import wandb
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
import matplotlib.pyplot as plt
from src.models.cifar_resnet import ResNet18
from src.dual_vars import DualVariableContainer
from src.utils import fix_randomness, get_git_root
from src.problems import GenericProblem
from src.solvers import GradientPDSolver


################################################################################
# SETUP PROBLEM
################################################################################

def subgroup_aggregated_classification_loss(model, x, y, g, loss_fn, num_groups):
    assert x.shape[0] == y.shape[0]==g.shape[0] and len(g.shape)==len(y.shape)==2 and y.shape[1]==g.shape[1]==1 
    # Flatten g to (B,)
    g = g.flatten()
    assert (g==y.flatten()).all() or num_groups==1
    assert torch.all((g >= 0) & (g < num_groups)), "Group indices out of range"
    losses = []
    out = model(x)
    for group in range(num_groups):
        mask = (g == group)
        if mask.any():
            # x_group = x[mask]
            y_group = y[mask]
            out_group = out[mask]
            # assert out.shape[0] == x_group.shape[0] == y_group.shape[0]
            assert y_group.shape[1]==1 and len(out.shape)==2
            loss = loss_fn(out_group, y_group.flatten())
        else:
            loss = torch.tensor(0.0, dtype=torch.float32, device=x.device)
        losses.append(loss)
    constraints = torch.stack(losses)
    assert constraints.shape == (num_groups,)
    return constraints

class VanillaClassification(GenericProblem):
    
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss() 
    
        self.add_constraint(
            lambda model, x, y, _: subgroup_aggregated_classification_loss(model, x, y, torch.zeros_like(y), self.loss_fn, num_groups=1),
            ctype="objective",
            in_indices=('x', 'y', 'g'),
            name="cross_entropy"
        )


class SubgroupClassification(GenericProblem):
    def __init__(self, num_groups=1, loss_fn = torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.num_groups = num_groups
        self.loss_fn = loss_fn
    
        self.add_constraint(
            lambda model: torch.tensor(0.0, dtype=torch.float32, requires_grad=False, device = model._mydevice),
            ctype="objective",
            in_indices=(),
            name="null_obj"
        )
            
        self.add_constraint(
            lambda model, x, y, g: subgroup_aggregated_classification_loss(model, x, y, g, self.loss_fn, num_groups=self.num_groups),
            ctype="vector",
            in_indices=("x","y","g"),
            name="subgroup_loss",
            isequality=True, 
            num_constraints=self.num_groups
        )

def setup_problem(args):
    # Create an instance of SubgroupClassification
    if args.problem=='vanilla':
        problem=VanillaClassification()
    elif args.problem=='subgroup':
        problem = SubgroupClassification(num_groups=args.num_groups)
    return problem


################################################################################
# SETUP DATA
################################################################################

class DictWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        y = torch.tensor([y], dtype=torch.long)
        return {"x": x, "y": y, "g" : y.clone()}


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_dict = DictWrapper(cifar100_training)
    cifar100_training_loader = DataLoader(
        cifar100_training_dict, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_dict = DictWrapper(cifar100_test)
    cifar100_test_loader = DataLoader(
        cifar100_test_dict, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return cifar100_test_loader

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

def load_dataset(args):
    cifar100_training_loader = get_training_dataloader(
        CIFAR100_TRAIN_MEAN,
        CIFAR100_TRAIN_STD,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        CIFAR100_TRAIN_MEAN,
        CIFAR100_TRAIN_STD,
        num_workers=args.num_workers,
        batch_size=args.eval_batch_size,
        shuffle=False
    )
    return cifar100_training_loader, cifar100_test_loader
    

################################################################################
# SETUP MODEL AND OPTIM
################################################################################

def setup_model_and_optim(problem, args, DEVICE):
    if DEVICE =="cuda":
        def dpwrapper(mod):
            newmod=torch.nn.DataParallel(mod)
            return newmod
        wrapper = dpwrapper
    else : 
        wrapper = lambda mod : mod
    # Load Resnet18
    model_func = ResNet18
    model = wrapper(model_func(num_classes=len(CIFARCLASSES))).to(DEVICE)
    model._mydevice=DEVICE
    
    # Setup optimizers
    primal_optim = torch.optim.Adam(
        model.parameters(),
        lr=args.primal_lr,
        weight_decay=args.primal_wd,
        betas=(args.primal_beta1, args.primal_beta2)
    )
    if args.primal_sgd_momentum : 
        primal_optim = torch.optim.SGD(
            model.parameters(),
            lr=args.primal_lr,
            weight_decay=args.primal_wd,
            nesterov=True, 
            momentum=args.sgd_momentum
        )
            
    primal_scheduler=None
    if args.cosine_schedule : 
        primal_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(primal_optim, T_max=args.num_epochs)
        
    if args.custom_schedule_1:
        primal_scheduler = torch.optim.lr_scheduler.MultiStepLR(primal_optim, milestones=[60, 120, 160], gamma=0.2)
    
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
    
    return model, duals, primal_optim, dual_optim, primal_scheduler


def log_models(args, run, solver):
    run_name = run.name
    run_id = run.id
    os.makedirs(args.output_dir, exist_ok=True)
    # Save and log primal model
    primal_path = os.path.join(args.output_dir, f"{run_name}_{run_id}_primal.pt")
    torch.save(solver.model.state_dict(), primal_path)
    step_str = solver.epoch_count if not solver.finished else "final"
    run.log_model(primal_path, name=f"{run_name}_{run_id}_{step_str}_primal")
    
################################################################################
# SETUP EVALUATORS AND CHECKPOINTS
################################################################################

def setup_evaluators(args, run, DEVICE):
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
        if solver.epoch_count == 0 and not args.save_first : 
            return
        if solver.finished and not args.save_final:
            return
        log_models(args, run, solver)
                    
    return [log_base_metrics, log_constraintwise_metrics, checkpoint_models]

################################################################################
# SETUP POST-TRAINING EVALUATION
################################################################################

def evaluate_model(model, evalloader, device="cpu"):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in evalloader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            assert x.shape==(y.shape[0], 3, 32, 32) and y.shape[1]==1
            y = y.view(-1)# (B,)

            outputs = model(x)  # (B,10)
            assert outputs.shape == (x.shape[0], len(CIFARCLASSES))
            preds = outputs.argmax(dim=1)  # predicted class

            all_preds.append(preds)
            all_targets.append(y)

    # concatenate everything
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()

    # compute accuracy
    acc = (all_preds == all_targets).mean()

    # confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(len(CIFARCLASSES)))
    return acc, cm


def evaluate(args, run, model, duals, testloader, total_step_count, DEVICE):
    acc, cm =evaluate_model(model, testloader, DEVICE)
    model.eval()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=CIFARCLASSES)
    disp.plot()
    disp.ax_.set_title("Final Eval (Test)")
    out=wandb.Image(disp.figure_)
    run.log({f"test/confmat_img" : out}, step=total_step_count)
    run.log({f"test/confmat" : cm.tolist()}, step=total_step_count)
    run.log({f"test/test_acc" : acc.item()}, step=total_step_count)
    
    for dv, _, ctype, _, cons_name, _ in duals.dual_blocks:
        if ctype == "vector":
            assert cons_name == "subgroup_loss" 
            dualvals=dv.eval()
            assert dualvals.shape==(args.num_groups, )
            # make a bar plot 
            plt.figure()
            plt.bar(np.arange(args.num_groups), dualvals.detach().cpu().numpy())
            plt.xlabel("Group")
            plt.ylabel("Dual Value")
            plt.title("Dual Variables per Group")
            bar_img = wandb.Image(plt.gcf())
            run.log({f"test/dualvals_bar": bar_img}, step=total_step_count)
            plt.close()
            
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
    model, duals, primal_optim, dual_optim, primal_scheduler = setup_model_and_optim(problem, args, DEVICE)
    with run as run:
        evaluators = setup_evaluators(args, run, DEVICE)
        solver = GradientPDSolver(problem, model, duals, primal_optim, dual_optim, trainloader, evaluators=evaluators, termination_criteria= [], device=DEVICE, primal_scheduler =primal_scheduler)
        
        model, duals = solver.train(args.num_epochs)
        
        evaluate(args, run, model, duals, testloader, solver.total_step_count, DEVICE)


################################################################################
# CLI ARGS AND MAIN()
################################################################################


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, choices=['vanilla', 'subgroup'], required=True, help="Problem type: 'vanilla' (standard CE) or 'subgroup' (group-wise constraints).")
    parser.add_argument('--num_groups', type=int, default=10, help="Number of groups for subgroup constraints.")
    
    # optimizers: primal
    parser.add_argument('--batch_size', type=int, default=139, help="Training batch size.")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--primal_lr', type=float, default=1e-3, help="Primal optimizer learning rate.")
    parser.add_argument('--primal_wd', type=float, default=0.0, help="Primal optimizer weight decay.")
    parser.add_argument('--primal_beta1', type=float, default=0.9, help="Beta1 for Adam (primal).")
    parser.add_argument('--primal_beta2', type=float, default=0.999, help="Beta2 for Adam (primal).")
    # sgd config
    parser.add_argument('--primal_sgd_momentum', action='store_true', default=False, help="Use SGD with momentum for primal updates.")
    parser.add_argument('--sgd_momentum', type=float, default=1e-3, help="Momentum value for SGD (if enabled).")
    parser.add_argument('--cosine_schedule', action='store_true', default=False, help="Use cosine annealing LR schedule for primal optimizer.")
    parser.add_argument('--custom_schedule_1', action='store_true', default=False, help="Use the predefined multistep LR schedule (60,120,160).")
    
    # optimizers: dual
    parser.add_argument('--use_dual_sgd', action='store_true', default=False, help="Use SGD for dual optimizer instead of Adam.")
    parser.add_argument('--dual_lr', type=float, default=1e-3, help="Dual optimizer learning rate.")
    parser.add_argument('--dual_wd', type=float, default=0.0, help="Dual optimizer weight decay.")
    parser.add_argument('--dual_beta1', type=float, default=0.9, help="Beta1 for Adam (dual).")
    parser.add_argument('--dual_beta2', type=float, default=0.999, help="Beta2 for Adam (dual).")
    
    # logging
    parser.add_argument('--logging_interval', type=int, default=1, help='Log metrics to wandb every N steps.')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Save checkpoints every N epochs.')
    parser.add_argument('--save_final', action='store_true', default=False, help="Save final model after training.")
    parser.add_argument('--save_first', action='store_true', default=False, help="Save model at epoch 0.")
    
    # wandb
    parser.add_argument('--entity', type=str, required=True, help='Wandb entity.')
    parser.add_argument('--project_name', type=str, default='reproduce', help='Wandb project name.')
    parser.add_argument('--wandbgrp', type=str, default='inter100', help='Wandb run group.')
    parser.add_argument('--namestr', type=str, default='', help='Optional name suffix for the run.')
    
    # Misc
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cpu or cuda).')
    parser.add_argument('--num_workers', type=int, default=0, help='Dataloader num_workers.')
    parser.add_argument('--eval_batch_size', type=int, default=139, help="Evaluation batch size.")
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory for checkpoints and outputs.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (None for random).')
    return parser

if __name__ == "__main__":
    parser = build_parser()
    args= parser.parse_args()
    CIFARCLASSES = [str(i) for i in range(1,101)]
    train(args)
