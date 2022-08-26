DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE

nohup python job_script.py --output rl_scratch_exp --config rl_scratch.txt  --record_video True --save_id $DATE --wandb_activate True --wandb_entity quantumiracle >> log/$DATE/rl_scratch.log &