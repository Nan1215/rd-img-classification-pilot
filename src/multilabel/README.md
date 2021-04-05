# Multilabel classification


Harvest data

`python src/multilabel/harvest_data.py --vocab_json vocabularies/vocabulary.json --n 3000 --saving_dir data/multilabel --reusability open`

Download images

`python src/download_images.py --csv_path data/multilabel/multilabel_dataset_open_permission.csv --saving_dir ../training_data_multilabel_open_permission --mode multi_label`

Train 

`python src/train_multilabel.py --data_dir ../training_data_multilabel_open_permission --annotations data/multilabel/multilabel_dataset_open_permission.csv --saving_dir ../results_multilabel_open_permission --input_size 128 --batch_size 64 --learning_rate 1e-5 --resnet_size 18 --max_epochs 100 --num_workers 4`

