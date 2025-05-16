sh finalizedata.sh
###############Training##############
sh trainingcommand.sh

###########inferencing###############
#option 1
# With explicit prompt
python scripts/inference_cli.py \
  --model_dir output/ll3-8b-ft \
  --prompt "Explain the MITRE ATT&CK framework." \
  --max_new_tokens 128 \
  --temperature 0.7 \
  --top_k 50 \
  --top_p 0.95

