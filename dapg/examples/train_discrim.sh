DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE

echo nohup python -W ignore train_discrim.py --itr 10000 --save_id $DATE --wandb_activate True --wandb_entity quantumiracle >> log/$DATE/discriminator.log
nohup python -W ignore train_discrim.py --itr 10000 --save_id $DATE --wandb_activate True --wandb_entity quantumiracle >> log/$DATE/discriminator.log &