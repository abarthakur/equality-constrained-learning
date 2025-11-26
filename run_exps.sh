

ENTITY=""
PROJECT_NAME="reproduce_exps"


##############################################################################

NUM_EPOCHS=200
# NUM_EPOCHS=1

# ERM, CIFAR-10
for seed in 0 1 2 3 4; do 
    python interpolate.py --seed ${seed} --entity ${ENTITY}  --project_name ${PROJECT_NAME} --wandbgrp cifar10_erm --namestr cifar10_erm_${seed} --problem vanilla --num_groups -1 --batch_size 128 --num_epochs ${NUM_EPOCHS}  --primal_lr 0.1 --cosine_schedule --primal_sgd_momentum --sgd_momentum 0.9 --primal_wd 5e-4 --dual_lr 0.0 --use_dual_sgd --eval_batch_size 512 --logging_interval 1 --checkpoint_interval 10000 --num_workers 4 --save_final;
done

# ECL, CIFAR-10
for seed in 0 1 2 3 4; do 
    python interpolate.py --seed ${seed} --entity ${ENTITY}  --project_name ${PROJECT_NAME} --wandbgrp cifar10_ecl --namestr cifar10_ecl_${seed} --problem subgroup --num_groups 10 --batch_size 128 --num_epochs ${NUM_EPOCHS}  --primal_lr 0.1 --cosine_schedule --primal_sgd_momentum --sgd_momentum 0.9 --primal_wd 5e-4 --dual_lr 1e-4 --eval_batch_size 512 --logging_interval 1 --checkpoint_interval 10000 --num_workers 4 --save_final;
done


# ERM, CIFAR-100
for seed in 0 1 2 3 4; do 
    python interpolate100.py --seed ${seed} --entity ${ENTITY} --project_name ${PROJECT_NAME} --wandbgrp cifar100_aug_erm --namestr cifar100_aug_erm_${seed} --problem vanilla --num_groups -1 --batch_size 128 --num_epochs ${NUM_EPOCHS}  --primal_lr 0.1 --custom_schedule_1 --primal_sgd_momentum --sgd_momentum 0.9 --primal_wd 5e-4 --dual_lr 0.0 --eval_batch_size 512 --logging_interval 1 --checkpoint_interval 10000 --num_workers 4 --save_final
done;

# ECL, ADAM, CIFAR 100
for seed in 0 1 2 3 4; do 
    python interpolate100.py --seed ${seed} --entity ${ENTITY} --project_name ${PROJECT_NAME} --wandbgrp cifar100_aug_ecl --namestr cifar100_aug_ecl_${seed} --problem subgroup --num_groups 100 --batch_size 128 --num_epochs ${NUM_EPOCHS}  --primal_lr 0.1 --custom_schedule_1 --primal_sgd_momentum --sgd_momentum 0.9 --primal_wd 5e-4 --dual_lr 1e-5 --eval_batch_size 512 --logging_interval 1 --checkpoint_interval 10000 --num_workers 4 --save_final
done;

##############################################################################

NUM_EPOCHS=300000
# NUM_EPOCHS=100

# PINN, Beta=30
for seed in {700..704}; do
    python convec.py --seed ${seed} --entity ${ENTITY} --project_name ${PROJECT_NAME} --wandbgrp pinn-30 --namestr pinn-30-${seed} --Co_beta 30.0 --ctypes obj obj obj --lambdas 100.0 100.0 1.0 --dual_lr 1e-4 --batch_size 1000 --num_epochs ${NUM_EPOCHS} --primal_layers 4 --primal_width 50 --primal_lr 1e-3 --schedule_lr --primal_lr_step 5000 --primal_lr_decay 0.9 --testfile solution_convection_30.0_512_251.npy --logging_interval 1 --evaluation_interval 1  --eval_batch_size 5000 --plotting_interval 1000000 --checkpoint_interval 10000000 --save_final 
done

# FDC, Beta=30
for seed in {700..704}; do
    python convec.py --seed ${seed} --entity ${ENTITY}  --project_name ${PROJECT_NAME} --wandbgrp fdc-30 --namestr fdc-30-${seed} --Co_beta 30.0 --ctypes vec vec vec --lambdas 1.0 1.0 1.0 --dual_lr 1e-4 --batch_size 1000 --num_epochs ${NUM_EPOCHS} --primal_layers 4 --primal_width 50 --primal_lr 1e-3 --schedule_lr --primal_lr_step 5000 --primal_lr_decay 0.9 --testfile solution_convection_30.0_512_251.npy --logging_interval 1 --evaluation_interval 1  --eval_batch_size 5000 --plotting_interval 1000000 --checkpoint_interval 10000000 --save_final;
done

# PINN, Beta=50
for seed in {700..704}; do
    python convec.py --seed ${seed} --entity ${ENTITY}  --project_name ${PROJECT_NAME} --wandbgrp pinn-50 --namestr pinn-50-${seed} --Co_beta 50.0 --ctypes obj obj obj --lambdas 100.0 100.0 1.0 --dual_lr 1e-4 --batch_size 1000 --num_epochs ${NUM_EPOCHS} --primal_layers 4 --primal_width 50 --primal_lr 1e-3 --schedule_lr --primal_lr_step 5000 --primal_lr_decay 0.9 --testfile solution_convection_50.0_512_251.npy --logging_interval 1 --evaluation_interval 1  --eval_batch_size 5000 --plotting_interval 1000000 --checkpoint_interval 10000000 --save_final 
done

# FDC, Beta=50
for seed in {700..704}; do
    python convec.py --seed ${seed} --entity ${ENTITY}  --project_name ${PROJECT_NAME} --wandbgrp fdc-50 --namestr fdc-50-${seed} --Co_beta 50.0 --ctypes vec vec vec --lambdas 1.0 1.0 1.0 --dual_lr 1e-4 --batch_size 1000 --num_epochs ${NUM_EPOCHS} --primal_layers 4 --primal_width 50 --primal_lr 1e-3 --schedule_lr --primal_lr_step 5000 --primal_lr_decay 0.9 --testfile solution_convection_50.0_512_251.npy --logging_interval 1 --evaluation_interval 1  --eval_batch_size 5000 --plotting_interval 1000000 --checkpoint_interval 10000000 --save_final
done

##############################################################################

NUM_EPOCHS=16000
# NUM_EPOCHS=100
TOLS="0.1 0.05 0.01 0.00001"

for i in {0..9}; do
    # Unconstrained
    python dempar.py --seed 0 --entity ${ENTITY} --project_name ${PROJECT_NAME} --wandbgrp dempar \
      --namestr uncons_ds${i} \
      --batch_size 5000 --num_epochs ${NUM_EPOCHS} --primal_lr 0.2 --dual_lr 1e-3 \
      --sigtemp 8.0 --dataset_file procdata_${i}.p --lagterm &

    # Equality
    python dempar.py --seed 0 --entity ${ENTITY} --project_name ${PROJECT_NAME} --wandbgrp dempar \
      --namestr eq_ds${i} \
      --cspecs "dp" --eqs "true" \
      --batch_size 5000 --num_epochs ${NUM_EPOCHS} --primal_lr 0.2 --dual_lr 1e-3 \
      --sigtemp 8.0 --dataset_file procdata_${i}.p --lagterm &

    # Inequality (loop over tolerances)
    for tol in ${TOLS}; do
        python dempar.py --seed 0 --entity ${ENTITY} --project_name ${PROJECT_NAME} --wandbgrp dempar \
          --namestr ineq_tol${tol}_ds${i} \
          --cspecs "dp" --eqs "false" --tols ${tol} \
          --batch_size 5000 --num_epochs ${NUM_EPOCHS} --primal_lr 0.2 --dual_lr 1e-3 \
          --sigtemp 8.0 --dataset_file procdata_${i}.p --lagterm &
    done
    wait
done

##############################################################################

WANDBGRP="prescriptive"
NUM_EPOCHS=16000
# NUM_EPOCHS=100
RHS="0.1 0.3 0.5 0.7 0.9"

for i in {0..9}; do
    # Unconstrained
    python dempar.py --seed 0 --entity ${ENTITY} --project_name ${PROJECT_NAME} --wandbgrp dempar \
      --namestr uncons_ds${i} \
      --batch_size 5000 --num_epochs ${NUM_EPOCHS} --primal_lr 0.2 --dual_lr 1e-3 \
      --sigtemp 8.0 --dataset_file procdata_${i}.p --lagterm &

    # Equality
    python dempar.py --seed 0 --entity ${ENTITY} --project_name ${PROJECT_NAME} --wandbgrp dempar \
      --namestr eq_ds${i} \
      --cspecs "dp" --eqs "true" \
      --batch_size 5000 --num_epochs ${NUM_EPOCHS} --primal_lr 0.2 --dual_lr 1e-3 \
      --sigtemp 8.0 --dataset_file procdata_${i}.p --lagterm &

    # Inequality (loop over tolerances)
    for rhs in ${RHS}; do
        python prescriptive.py --seed 0 --project_name ${PROJECT} --wandbgrp ${WANDBGRP} \
          --namestr presc_rhs${rhs}_ds${i} --rhs ${rhs} \
          --batch_size 5000 --num_epochs ${NUM_EPOCHS} --primal_lr 0.2 --dual_lr 1e-3 \
          --sigtemp 8.0 --dataset_file procdata_${i}.p --lagterm &
    done
    wait
done