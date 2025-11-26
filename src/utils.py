import subprocess
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import torch
import random
import time
import copy

import wandb

def get_git_root():
    return subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')


def fix_randomness(seed, cudnn_deterministic=False):
    seed = int(time.time()) if seed is None else seed
    # set random seeds for Python, numpy and pytorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if cudnn_deterministic: 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

def log_pde_solution(run, step_count, grid_dim, inputs, solution_array, wandb_name):
    """
    Logs PDE solution to wandb as a line plot (1D) or heatmap (2D).

    Args:
        run: wandb run object.
        step_count: Current step or epoch.
        grid_dim: 1 for 1D, 2 for 2D.
        inputs: dict of input arrays (e.g., {'x': x} or {'x': x, 'y': y}).
        solution_array: Solution values, shape should match grid.
        wandb_name: Name for wandb log.
    """
    matplotlib.use('agg')
    if grid_dim == 1:
        # Assume inputs has one key, e.g., 'x'
        x = list(inputs.values())[0].cpu().detach().numpy()
        plt.figure()
        plt.plot(x, solution_array.cpu().detach().numpy())
        plt.xlabel(list(inputs.keys())[0])
        plt.ylabel("Solution")
        plt.title(f"Step {step_count}")
        run.log({wandb_name : plt}, step=step_count)

    elif grid_dim == 2:
        # Assume inputs has two keys, e.g., 'x' and 'y'
        x_key, y_key = list(inputs.keys())
        x = inputs[x_key].cpu().detach().numpy()
        y = inputs[y_key].cpu().detach().numpy()
        # Check if x and y define an equally spaced grid
        if not (np.allclose(np.diff(x), np.diff(x)[0]) and np.allclose(np.diff(y), np.diff(y)[0])):
            print("Warning: x or y are not equally spaced.")
        # Assume solution_array is shape (len(y), len(x)) or flatten and reshape
        if solution_array.ndim == 1:
            solution_array = solution_array.reshape(len(y), len(x))
        plt.figure()
        extent = [x.min(), x.max(), y.min(), y.max()]
        plt.imshow(solution_array.detach().cpu().numpy(), extent=extent, origin='lower', aspect='auto')
        plt.xlabel(x_key)
        plt.ylabel(y_key)
        plt.title(f"Step {step_count}")
        plt.colorbar(label="Solution")
        run.log({wandb_name : plt}, step=step_count)

    else:
        print(f"Warning: grid_dim={grid_dim} not supported for logging.")
        
        
def get_diagnostic_plot(step_count, inputs, solution_array):
    """
    Creates and returns a diagnostic plot (line for 1D, heatmap for 2D) for PDE solution.

    Args:
        run: wandb run object.
        step_count: Current step or epoch.
        inputs: dict of input arrays (e.g., {'x': x} or {'x': x, 'y': y}).
        solution_array: Solution values, shape should match grid.
        wandb_name: Name for wandb log.

    Returns:
        fig: The matplotlib figure object.
    """
    matplotlib.use('agg')

    input_keys = list(inputs.keys())
    input_arrays = [inputs[k].cpu().detach().numpy() for k in input_keys]
    grid_dim = len(input_arrays)

    if grid_dim == 1:
        x = np.asarray(input_arrays[0]).squeeze()
        y = solution_array
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        fig, ax = plt.subplots()
        ax.plot(x_sorted, y_sorted)
        ax.set_xlabel(input_keys[0])
        ax.set_ylabel("Output")
        ax.set_title(f"Step {step_count}")
        out = wandb.Image(fig)
        plt.close(fig)
        return out

    elif grid_dim == 2:
        x = np.asarray(input_arrays[0]).flatten()
        y = np.asarray(input_arrays[1]).flatten()
        sol = solution_array

        x_unique = np.unique(x)
        y_unique = np.unique(y)
        nx, ny = len(x_unique), len(y_unique)

        # Check that the inputs form a full grid
        if x.shape[0] != nx * ny or y.shape[0] != nx * ny:
            raise ValueError(f"Input arrays do not represent a full grid: expected {nx * ny} points, got {x.shape[0]}.")

        # Try to reshape assuming row-major (y changes faster)
        try:
            x_grid, y_grid = np.meshgrid(x_unique, y_unique, indexing='xy')
            x_expected = x_grid.ravel()
            y_expected = y_grid.ravel()

            if not (np.allclose(x, x_expected) and np.allclose(y, y_expected)):
                raise ValueError("Input arrays are not ordered as row-major full grid (ravel of meshgrid with indexing='xy').")

            sol_grid = sol.reshape(ny, nx)  # shape should match y-rows, x-cols
        except Exception as e:
            raise ValueError(f"Failed to reshape or validate grid order: {e}")

        fig, ax = plt.subplots()
        extent = [x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()]
        im = ax.imshow(sol_grid, extent=extent, origin='lower', aspect='auto')
        ax.set_xlabel(input_keys[0])
        ax.set_ylabel(input_keys[1])
        ax.set_title(f"Step {step_count}")
        fig.colorbar(im, ax=ax, label="Solution")
        out = wandb.Image(fig)
        plt.close(fig)
        return out

    else:
        print(f"Warning: grid_dim={grid_dim} not supported for logging.")
        return None


def get_histogram(step_count, solution_array, bin_count=50):
    """
    Creates a histogram plot of the solution values.

    Args:
        step_count: Current step or epoch (for labeling).
        solution_array: Tensor of solution values.

    Returns:
        fig: The matplotlib figure object.
    """
    matplotlib.use('agg')
    sol = solution_array.flatten()

    fig, ax = plt.subplots()
    try :
        ax.hist(sol, bins=bin_count, edgecolor='black')
    except ValueError :
        plt.close(fig)
        if bin_count > 1 : 
            return get_histogram(step_count, solution_array, bin_count=max(bin_count-10, 1))
        else: 
            # make an empty plot
            pass
        
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Step {step_count}")
    out = wandb.Image(fig)
    plt.close(fig)
    return out
