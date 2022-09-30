DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE

declare -a envs=('relocate-v0' 'pen-v0' 'door-v0' 'hammer-v0')

for i in ${!envs[@]}; do
    echo nohup python -W ignore train_discrim.py --leave_one_out ${envs[$i]} --itr 10000 --save_id $DATE --wandb_activate True --wandb_entity quantumiracle >> log/$DATE/discriminator.log
    nohup python -W ignore train_discrim.py --leave_one_out ${envs[$i]} --itr 10000 --save_id $DATE --wandb_activate True --wandb_entity quantumiracle >> log/$DATE/discriminator.log &
done