# Setup path to include the source directory
import subprocess
import sys
import os

result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True, check=True)
gitroot = result.stdout.strip()
sys.path.append(gitroot)
sys.path.append(os.path.join(gitroot, "external"))

import argparse
import numpy as np

from src.utils import get_git_root
# from edgar_reference_equations.poissons import exact_solution 

# def growth_solution(r, y0, t):
#     return y0 * np.exp(r * t)

# taken from https://github.com/arkadaw9/r3_sampling_icml2023/blob/main/Convection/systems.py
def convection_diffusion_solution(u0, nu, beta, source=0, xgrid=256, nt=100):
    """Calculate the u solution for convection/diffusion, assuming PBCs.
    Args:
        u0: Initial condition
        nu: viscosity coefficient
        beta: wavespeed coefficient
        source: q (forcing term), option to have this be a constant
        xgrid: size of the x grid
    Returns:
        u_vals: solution
    """

    N = xgrid
    h = 2 * np.pi / N
    x = np.arange(0, 2*np.pi, h) # not inclusive of the last point
    t = np.linspace(0, 1, nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)

    # call u0 this way so array is (n, ), so each row of u should also be (n, )
    # u0 = function(u0)
    # simply assume we pass the right u_0
    u0 = u0(x)

    G = (np.copy(u0)*0)+source # G is the same size as u0

    IKX_pos =1j * np.arange(0, N/2+1, 1)
    IKX_neg = 1j * np.arange(-N/2+1, 0, 1)
    IKX = np.concatenate((IKX_pos, IKX_neg))
    IKX2 = IKX * IKX

    uhat0 = np.fft.fft(u0)
    nu_factor = np.exp(nu * IKX2 * T - beta * IKX * T)
    A = uhat0 - np.fft.fft(G)*0 # at t=0, second term goes away
    uhat = A*nu_factor + np.fft.fft(G)*T # for constant, fft(p) dt = fft(p)*T
    u = np.real(np.fft.ifft(uhat))

    u_vals = u.flatten()
    return u_vals, X, T

def main(args):
    # if args.problem == "growth":
    #     if args.Gr_r is None or args.Gr_y0 is None:
    #         raise ValueError("For growth, --Gr_r and --Gr_y0 must be specified.")
    #     # Bounds for growth: t in [0, 1.0]
    #     t_splits = args.grid_spec[0]
    #     t_grid = np.linspace(0, 1.0, t_splits + 1)
    #     solution = growth_solution(args.Gr_r, args.Gr_y0, t_grid)
    #     # Prepare save path
    #     gitroot = get_git_root()
    #     save_dir = os.path.join(gitroot, "data", "pdelearn", "growth")
    #     os.makedirs(save_dir, exist_ok=True)
    #     grid_spec_str = "_".join(str(x) for x in args.grid_spec)
    #     filename = f"{args.prefix + "_" if args.prefix!="" else ""}solution_{args.Gr_r}_{args.Gr_y0}_{grid_spec_str}.npy"
    #     save_path = os.path.join(save_dir, filename)
    #     np.save(save_path, {"grid": {"t" : t_grid.reshape(-1,1)}, "solution": solution.reshape(-1, 1)})
    #     print(f"Saved growth solution to {save_path}")
        
    # if args.problem == "poisson":
    #     if args.Po_kx is None or args.Po_ky is None:
    #         raise ValueError("For poisson, --Po_kx and --Po_ky must be specified.")
    #     # Bounds for Poisson: x, y in [0, 1.0]
    #     x_splits, y_splits = args.grid_spec
    #     x_grid = np.linspace(0, 1.0, x_splits + 1)
    #     y_grid = np.linspace(0, 1.0, y_splits + 1)
    #     xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')
    #     solution = exact_solution(xx, yy, args.Po_kx, args.Po_ky)
    #     # Prepare save path
    #     gitroot = get_git_root()
    #     save_dir = os.path.join(gitroot, "data", "pdelearn", "poisson")
    #     os.makedirs(save_dir, exist_ok=True)
    #     grid_spec_str = "_".join(str(x) for x in args.grid_spec)
    #     filename = f"{args.prefix + '_' if args.prefix != '' else ''}solution_{args.Po_kx}_{args.Po_ky}_{grid_spec_str}.npy"
    #     save_path = os.path.join(save_dir, filename)
    #     np.save(save_path, {
    #         "grid": {"x": np.asarray(xx.reshape(-1, 1)), "y": np.asarray(yy.reshape(-1, 1))},
    #         "solution": np.asarray(solution.reshape(-1, 1))
    #     })
    #     print(f"Saved poisson solution to {save_path}")
        
    if args.problem == "convection":
        solution, xgrid, tgrid = convection_diffusion_solution(np.sin, nu=0, beta=args.Co_beta, xgrid=args.grid_spec[0], nt=args.grid_spec[1])
        # Prepare save path
        gitroot = get_git_root()
        save_dir = os.path.join(gitroot, "data", "pdelearn", "convection")
        os.makedirs(save_dir, exist_ok=True)
        grid_spec_str = "_".join(str(x) for x in args.grid_spec)
        filename = f"{args.prefix + '_' if args.prefix != '' else ''}solution_convection_{args.Co_beta}_{grid_spec_str}.npy"
        save_path = os.path.join(save_dir, filename)
        np.save(save_path, {
            "grid": {"x": xgrid.reshape(-1, 1), "t" : tgrid.reshape(-1,1) },
            "solution": solution.reshape(-1, 1)
        })
        print(f"Saved convection solution to {save_path}")

if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description="Generate PDE ground truths.")
    parser.add_argument("--problem", choices=["growth", "poisson", "allencahn", "convection"], required=True)
    parser.add_argument("--grid_spec", nargs='+', type=int, required=True, help="List of splits per dimension")
    parser.add_argument("--Gr_r", type=float, help="Growth equation parameter r")
    parser.add_argument("--Gr_y0", type=float, help="Growth equation initial value y0")
    parser.add_argument("--Po_kx", type=float, help="Poisson equation parameter kx")
    parser.add_argument("--Po_ky", type=float, help="Poisson equation parameter ky")
    parser.add_argument("--Co_beta", type=float, help="Diffusion equation parameter beta")
    parser.add_argument("--prefix", type=str, default="", help="Optional prefix for the saved filename")
    args = parser.parse_args()
    main(args)