import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
import numpy as np 

from utils.Trainer import Trainer_cls
from opts import parser


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

args = parser.parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(10)
torch.cuda.manual_seed(10)
np.random.seed(10)

def main():
    global args, device
       
    if args.model == 'Res3D':
        import models.ResI3D.config as config

    model = config.model
    model.to(device)
    train_loader = config.train_loader
    val_loader = config.val_loader

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2)

    
    print('prepare finished, start training')
    
    # train
    trainer = Trainer_cls(
        model,
        criterion,
        optimizer,
        train_loader,
        device,
        None        
    )

    trainer.train(
        args.epochs,
        test_loader=val_loader,
        loader_fn=None,
        lr_scheduler=lr_schedular,
        scheduler_metric='best_val_acc',
        bn_scheduler=None,
        saved_path='./saved_model',
        val_interval=args.val_interval
    )
    
if __name__ == '__main__':
    main()
