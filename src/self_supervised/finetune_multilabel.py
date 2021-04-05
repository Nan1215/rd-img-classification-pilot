from pathlib import Path
import torch
import torch.nn as nn
import os
import lightly
import json

ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

import sys

sys.path.append(ROOT_DIR)

from multilabel.train_multilabel import *

class FineTuneModel(nn.Module):
    def __init__(self, model,num_ftrs,output_dim):
        super().__init__()
        
        self.net = model
        self.fc1 = nn.Linear(num_ftrs, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y_hat = self.net.backbone(x).squeeze()
        y_hat = self.fc1(y_hat)
        y_hat = self.relu(y_hat)
        y_hat = self.fc2(y_hat)
        y_hat = self.sigmoid(y_hat)
        return y_hat
            
def load_simcrl(simclr_results_path,n_categories,device,model_size = 18):
    
    #load config
    conf_path = simclr_results_path.joinpath('conf.json')
    with open(conf_path,'r') as f:
        conf = json.load(f)

    #load model
    model_path = simclr_results_path.joinpath('checkpoint.pth')

    num_ftrs = conf['num_ftrs']
    
    model_name = conf['model_name']

    resnet = lightly.models.ResNetGenerator('resnet-'+str(model_size))
    last_conv_channels = list(resnet.children())[-1].in_features
    backbone = nn.Sequential(
        *list(resnet.children())[:-1],
        nn.Conv2d(last_conv_channels, num_ftrs, 1),
        nn.AdaptiveAvgPool2d(1)
    )

    if model_name == 'simclr':
        model = lightly.models.SimCLR(backbone, num_ftrs=num_ftrs)
    elif model_name == 'moco':
        model = lightly.models.MoCo(backbone, num_ftrs=num_ftrs, m=0.99, batch_shuffle=True)
        

    encoder = lightly.embedding.SelfSupervisedEmbedding(
        model,
        None,
        None,
        None
    )

    encoder.model.load_state_dict(torch.load(model_path))
    teacher = FineTuneModel(encoder.model,num_ftrs,n_categories).to(device)
    return teacher


def save_metrics(metrics,path):
  with open(path,'w') as f:
    json.dump(metrics,f)


def main(**kwargs):
  max_epochs = kwargs.get('max_epochs',100)
  annotations = kwargs.get('annotations')
  pretrained_dir = kwargs.get('pretrained_dir')
  data_dir = kwargs.get('data_dir')
  saving_dir = kwargs.get('saving_dir')
  input_size = kwargs.get('input_size',64)
  batch_size = kwargs.get('batch_size',32)
  num_workers = kwargs.get('num_workers',8)
  learning_rate = kwargs.get('learning_rate',1e-5)

  data_dir = Path(data_dir)
  df_path = Path(annotations)
  pretrained_dir = Path(pretrained_dir)
  saving_dir = Path(saving_dir)
  saving_dir.mkdir(parents=True, exist_ok=True)

  df = pd.read_csv(df_path)
  #filter images in df contained in data_path
  imgs_list = list(data_dir.iterdir())
  df['filepath'] = df['ID'].apply(lambda x:data_dir.joinpath(id_to_filename(x)+'.jpg'))
  df = df.loc[df['filepath'].apply(lambda x: Path(x) in imgs_list)]
  df['n_labels'] = df['category'].apply(lambda x: len(x.split()))
  df = df.sort_values(by='n_labels',ascending=False)
  df = df.drop_duplicates()
  print(df.shape)

  mlb = sklearn.preprocessing.MultiLabelBinarizer()

  imgs = np.array([str(path) for path in df['filepath'].values])

  labels = [item.split() for item in df['category'].values]
  labels = mlb.fit_transform(labels)

  class_index_dict = {i:c for i,c in enumerate(mlb.classes_)}

  #train test split
  imgs_train,imgs_evaluation,labels_train,labels_evaluation = train_test_split(imgs,labels,test_size = 0.3)
  imgs_val,imgs_test,labels_val,labels_test = train_test_split(imgs_evaluation,labels_evaluation,test_size = 0.5)

  train_transform = transforms.Compose([
      transforms.Resize((input_size, input_size)),
      transforms.RandomHorizontalFlip(0.5),
      transforms.RandomVerticalFlip(0.5),
      transforms.ToTensor(),
      # this normalization is required https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])


  test_transform = transforms.Compose([
      transforms.Resize((input_size, input_size)),
      transforms.ToTensor(),
      # this normalization is required https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  trainset = MultilabelDataset(imgs_train,labels_train,transform = train_transform)
  trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=num_workers,drop_last=True)

  valset = MultilabelDataset(imgs_val,labels_val,transform = test_transform)
  valloader = DataLoader(valset, batch_size=batch_size,shuffle=True, num_workers=num_workers,drop_last=True)

  testset = MultilabelDataset(imgs_test,labels_test,transform = test_transform)
  testloader = DataLoader(testset, batch_size=batch_size,shuffle=True, num_workers=num_workers,drop_last=True)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  n_categories = labels.shape[1]

  model = load_simcrl(pretrained_dir,n_categories,device,model_size = 34)

  loss_function = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=1.0)

  model = train(
      model = model,
      trainloader = trainloader,
      valloader = valloader,
      device = device,
      loss_function = loss_function,
      optimizer = optimizer,
      scheduler = scheduler,
      max_epochs = max_epochs,
      saving_dir = saving_dir,
  )


  test_metrics = evaluate(
      model = model,
      dataloader = testloader,
      loss_function = loss_function,
      device = device
  )
  print('Test')
  print_metrics(test_metrics)

  save_metrics(test_metrics,saving_dir.joinpath('test_metrics.json'))

  return 




if __name__ == '__main__':

 
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--pretrained_dir', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--saving_dir', required=False)
    parser.add_argument('--learning_rate', required=False)
    parser.add_argument('--max_epochs', required=False)
    parser.add_argument('--resnet_size', required=False)
    parser.add_argument('--num_workers', required=False)
    parser.add_argument('--batch_size', required=False)
    parser.add_argument('--input_size', required=False)

    args = parser.parse_args()


    if not args.saving_dir:
      saving_dir = ROOT_DIR
    else:
      saving_dir = args.saving_dir

    if not args.learning_rate:
      learning_rate = 0.00001
    else:
      learning_rate = float(args.learning_rate)

    if not args.max_epochs:
      max_epochs = 100
    else:
      max_epochs = int(args.max_epochs)

    if not args.resnet_size:
      resnet_size = 34
    else:
      resnet_size = int(args.resnet_size)


    if not args.num_workers:
      num_workers = 8
    else:
      num_workers = int(args.num_workers)

    if not args.batch_size:
      batch_size = 32
    else:
      batch_size = int(args.batch_size)



    main(
        data_dir = args.data_dir ,
        annotations = args.annotations,
        pretrained_dir = args.pretrained_dir,
        saving_dir = saving_dir,
        learning_rate = learning_rate,
        max_epochs = max_epochs,
        resnet_size = resnet_size,
        num_workers = num_workers,
        batch_size = batch_size,
    )
