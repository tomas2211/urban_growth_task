#!/bin/bash

#python eval_net.py --checkpoint_path models/baseline_000049.pth --out_folder model_eval/baseline
#python eval_net.py --checkpoint_path models/bs32_000049.pth --out_folder model_eval/bs32
#python eval_net.py --checkpoint_path models/bs64_000049.pth --out_folder model_eval/bs64
#python eval_net.py --checkpoint_path models/drop0_000049.pth --out_folder model_eval/drop_0
#python eval_net.py --checkpoint_path models/drop0-5_000049.pth --out_folder model_eval/drop_0-5
#python eval_net.py --checkpoint_path models/labweight_000049.pth --out_folder model_eval/lab_weight
#python eval_net.py --checkpoint_path models/labweight-weak_000049.pth --out_folder model_eval/lab_weight_weak

python create_timeseries.py --checkpoint_path models/drop0-5_000049.pth --out_folder model_eval/drop_0-5