DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE

for i in ${!methods[@]}; do
    for j in ${!envs[@]}; do
        nohup python -W ignore train_discrim.py --itr 100000 --save_id $DATE --wandb_activate True --wandb_entity quantumiracle >> log/$DATE/discriminator.log &
    done
done