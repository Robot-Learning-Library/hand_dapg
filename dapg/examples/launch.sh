DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE

declare -a tasks=('rl_scratch' 'bcrl' 'dapg')

for i in ${!tasks[@]}; do
    nohup python job_script.py --output results/$DATE/${tasks[$i]}_exp --config ${tasks[$i]}.txt  --record_video True --save_id $DATE --wandb_activate True --wandb_entity quantumiracle >> log/$DATE/${tasks[$i]}.log &
done