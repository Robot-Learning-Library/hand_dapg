DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE
mkdir -p results/$DATE
# declare -a methods=('rl_scratch' 'bcrl' 'dapg' 'npg_discrim')
# declare -a envs=('relocate' 'pen' 'hammer' 'door')
declare -a methods=('rl_scratch')
declare -a envs=('relocate')


for i in ${!methods[@]}; do
    for j in ${!envs[@]}; do
        nohup python -W ignore job_script.py --output results/$DATE/${methods[$i]}_${envs[$j]}_exp --config cfg/${methods[$i]}_${envs[$j]}.txt  --record_video --save_id $DATE --wandb_activate --wandb_entity quantumiracle >> log/$DATE/${methods[$i]}_${envs[$j]}.log &
    done
done