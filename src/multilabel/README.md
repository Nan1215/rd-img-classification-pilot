# Multilabel classification


Harvest data

`python src/multilabel/harvest_data.py --vocab_json vocabularies/vocabulary.json --n 3000 --saving_dir data/multilabel --reusability open`

Download images

`python src/download_images.py --csv_path data/dataset.csv --saving_dir ../training_data_multilabel --mode multi_label`

Train 

`python src/multilabel/train_multilabel.py --data_dir ../training_data_multilabel --annotations multilabel_dataset.csv  --epochs 100 --patience 10 --experiment_name multilabel_training --img_aug 0.5 --saving_dir ../ `

