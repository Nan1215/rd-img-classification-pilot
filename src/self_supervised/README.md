

Train self supervised model

`python src/self_supervised/train.py --model_name moco --model_size 34 --data_path ../training_data_3000_open_permission --max_epochs 200 --saving_dir ../self_supervised_results  --batch_size 64 --input_size 64 --num_ftrs 512 --hf_prob 0.5 --vf_prob 0.5 --experiment_name moco_34`

Fine tune multilabel

`python src/self_supervised/finetune_multilabel.py --data_dir ../training_data_multilabel --annotations data/multilabel/multilabel_dataset.csv --saving_dir ../results_finetuning_multilabel --pretrained_dir /home/jcejudo/self_supervised_results/simclr_34  --input_size 64 --batch_size 64 --learning_rate 1e-5 --max_epochs 100 --num_workers 8`
