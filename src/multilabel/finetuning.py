from pathlib import Path
import torch
import torch.nn as nn

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




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_dir = Path('/home/jcejudo/self_supervised_results/moco_50')

n_categories = 
teacher = load_simcrl(pretrained_dir,n_categories,device,model_size = 50)
