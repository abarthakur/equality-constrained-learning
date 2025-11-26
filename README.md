# Learning with Statistical Equality Constraints

This is the official codebase accompanying the paper - [Learning with Statistical Equality Constraints](https://arxiv.org/abs/2511.14320) (Neurips 2025).


## Install

Clone the repository and install the required packages from the requirements file after activating your python (3.9) environment (recommended to use a virtual environment using `venv` or `conda`!). 

```bash
git clone https://github.com/abarthakur/equality-constrained-learning.git
cd ecl3
pip install -r requirements.txt
```

## Prepare data and run experiments

Running the following script will download and process the COMPAS dataset, and generate the convection solutions. 
```
bash prepare_data.sh
```

The following scripts correspond to the experiments in the paper : 
1. `dempar.py` : Demographic parity on COMPAS. 
2. `prescriptive.py` : Prescriptive rates on COMPAS. 
3. `convec.py` : Convection PDE learning. 
4. `interpolate.py` : Interpolation on CIFAR-10.
5. `interpolate100.py` : Interpolation on CIFAR-100.

Each script logs metrics, summary metrics, and models, to [wandb](https://wandb.ai/). The metrics/models can be thereafter retrieved using the wandb API for further analysis. You will need a wandb account to run these scripts. 

To run the experiments, modify the script `run_exps.sh`, specifically replacing the lines 
```
ENTITY=""
PROJECT_NAME=""
```
with your wandb entity (user/team), and your preferred wandb project. Thereafter, log in to wandb on your console (`wandb login`) and simply run 
```
bash run_exps.sh
```
to populate the project PROJECT_NAME with the runs required to replicate the figures and tables in the paper. 

## Citation

```
@misc{barthakur2025learningstatisticalequalityconstraints,
      title={Learning with Statistical Equality Constraints}, 
      author={Aneesh Barthakur and Luiz F. O. Chamon},
      year={2025},
      booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems},
}
```


```
@misc{barthakur2025learningstatisticalequalityconstraints,
      title={Learning with Statistical Equality Constraints}, 
      author={Aneesh Barthakur and Luiz F. O. Chamon},
      year={2025},
      eprint={2511.14320},
      archivePrefix={arXiv},
}
```