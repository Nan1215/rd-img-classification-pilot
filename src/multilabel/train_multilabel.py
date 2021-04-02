import argparse
import os

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd
from pathlib import Path
import sklearn
from sklearn.model_selection import train_test_split
import torch.optim as optim

ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
import sys
sys.path.append(os.path.join(ROOT_DIR))

from gradcam import *
from torch_utils import *

def id_to_filename(id):
    return id.replace('/','[ph]')

class MultilabelResNet(nn.Module):
    def __init__(self,size, output_size):
        super(MultilabelResNet, self).__init__()

        if size not in [18,34,50,101,152]:
            raise Exception('Wrong size for resnet')
        if size == 18:
            self.net = torchvision.models.resnet18(pretrained=True)
        elif size == 34:
            self.net = torchvision.models.resnet34(pretrained=True)
        elif size == 50:
            self.net = torchvision.models.resnet50(pretrained=True)
        elif size == 101:
            self.net = torchvision.models.resnet101(pretrained=True)
        elif size == 152:
            self.net = torchvision.models.resnet152(pretrained=True)

        #initialize the fully connected layer
        self.net.fc = nn.Linear(self.net.fc.in_features, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.net(x)
        out = self.sigmoid(out)
        return out
    
class MultilabelDataset(Dataset):
    """
    Pytorch training dataset class
    X: Numpy array containing the paths to the images
    y: Numpy array with the encoded labels
    returns batches of images, labels and images paths
    """
    def __init__(self, X, y, transform=None):
        self.transform = transform
        self.X = X
        self.y = y        

    def __len__(self):
        return self.y.shape[0] 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.X[idx]
        img, label = Image.open(img_path).convert('RGB'), self.y[idx]
        if self.transform:
            img = self.transform(img)

        return img,label,img_path


def print_metrics(metrics):
    loss = metrics['loss']
    coverage = metrics['coverage']
    lrap = metrics['lrap']
    label_ranking_loss = metrics['label_ranking_loss']
    print(f'loss:{loss:.3f} coverage:{coverage:.3f} lrap:{lrap:.3f} label_ranking_loss:{label_ranking_loss:.3f}')

def evaluate(**kwargs):
    
    model = kwargs.get('model')
    dataloader = kwargs.get('dataloader')
    loss_function = kwargs.get('loss_function')
    device = kwargs.get('device')
    
    ground_truth = []
    predictions = []
    
    model.eval()
    val_loss = 0.0
    for inputs,labels,_ in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        labels = labels.type_as(output)
        loss = loss_function(output, labels)
        val_loss += loss.item()
                
        ground_truth += list(labels.cpu().detach().numpy())
        predictions += list(output.cpu().detach().numpy())
        
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
        
    val_loss /= len(dataloader.dataset)
    coverage = sklearn.metrics.coverage_error(ground_truth, predictions)
    lrap = sklearn.metrics.label_ranking_average_precision_score(ground_truth, predictions)
    label_ranking_loss = sklearn.metrics.label_ranking_loss(ground_truth, predictions)
    
    return {'loss':val_loss,'coverage':coverage,'lrap':lrap,'label_ranking_loss':label_ranking_loss}


def train(**kwargs):
    model = kwargs.get('model')
    trainloader = kwargs.get('trainloader')
    valloader = kwargs.get('valloader')
    device = kwargs.get('device')
    optimizer = kwargs.get('optimizer')
    loss_function = kwargs.get('loss_function')
    saving_dir = kwargs.get('saving_dir')
    max_epochs = kwargs.get('max_epochs',100)
    patience = kwargs.get('patience',3)

    model.train()

    count = 0
    best_loss = 1e9
    for epoch in range(max_epochs):
        train_loss = 0.0

        
        # loop over batches
        for i, (inputs,labels,_) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()        
            output = model(inputs)
            labels = labels.type_as(output)
            loss = loss_function(output, labels)
            #backpropagate and update
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(trainloader.dataset)

        val_metrics = evaluate(
            model = model,
            dataloader = valloader,
            loss_function = loss_function,
            device = device
        )
        print_metrics(val_metrics)
        
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            best_metrics = val_metrics
            count = 0
            torch.save(model.state_dict(),saving_dir.joinpath(f'checkpoint.pth'))
        else:
            count += 1
        if count > patience:
            break

    model.load_state_dict(torch.load(saving_dir.joinpath(f'checkpoint.pth')))

    return model


def main(**kwargs):
  max_epochs = kwargs.get('max_epochs',50)
  annotations = kwargs.get('annotations')
  data_dir = kwargs.get('data_dir')
  saving_dir = kwargs.get('saving_dir')
  input_size = kwargs.get('input_size',64)
  batch_size = kwargs.get('batch_size',32)
  num_workers = kwargs.get('num_workers',2)
  learning_rate = kwargs.get('learning_rate',0.00001)
  num_workers = kwargs.get('max_epochs',100)

  data_dir = Path(data_dir)
  df_path = Path(annotations)
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
  model = MultilabelResNet(18,labels.shape[1]).to(device)

  loss_function = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  model = train(
      model = model,
      trainloader = trainloader,
      valloader = valloader,
      device = device,
      loss_function = loss_function,
      optimizer = optimizer,
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



    

  return 


if __name__ == '__main__':

    """
    Script for training crossvalidation

    Usage:

      python src/train.py --data_dir training_data --epochs 100 --patience 10 --experiment_name model_training --img_aug 0.5

    Parameters:

      data_dir: directory containing the dataset organized in 
                subdirectories for the different categories
                Required

      saving_dir: directory for saving the training results. 
                  If not specified this will be the root path of the repository

      experiment_name: tag for the results
                      Default: results_training

      learning_rate: Default: 0.00001

      epochs: Default: 100

      patience: Number of epochs for early stopping
                Default: 10

      resnet_size: size of the model. The allowed sizes are 18,34,50,101,152
                   Default: 34

      num_workers: Default: 4

      batch_size: Default: 64

      weighted_loss: Whether to penalize missclassifications for the minority classes
                     Recommended for imbalanced datasets.
                     Default: True

      img_aug: Whether to use image augmentation
                Default: True

    """

    ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
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
      num_workers = 4
    else:
      num_workers = int(args.num_workers)

    if not args.batch_size:
      batch_size = 64
    else:
      batch_size = int(args.batch_size)



    main(
        data_dir = args.data_dir ,
        annotations = args.annotations,
        saving_dir = saving_dir,
        learning_rate = learning_rate,
        max_epochs = max_epochs,
        resnet_size = resnet_size,
        num_workers = num_workers,
        batch_size = batch_size,
    )




        












