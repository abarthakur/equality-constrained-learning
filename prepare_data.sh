mkdir -p data/fairness
cd data/fairness
git clone https://github.com/propublica/compas-analysis.git
cd ../..

cd src/datasets
for seed in {0..9}; do python gen_compas_splits.py --seed $seed; done

python gen_convec.py --problem convection --grid_spec 512 251 --Co_beta 30.0 
python gen_convec.py --problem convection --grid_spec 512 251 --Co_beta 50.0 
