import sys
import os
from helper import get_source_dir, get_project_dir
sys.path.append(get_source_dir())

import argparse
import global_vars
import torch
import torch.nn as nn
import torch.optim as optim
from features.build_features import mnist
from model import MyAwesomeModel

def train(lr : float):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    model.cuda()
    dataset, _ = mnist()
    train_set, val_set = torch.utils.data.random_split(dataset, [20000, 5000])
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True)  
    valloader = torch.utils.data.DataLoader(val_set, batch_size=64)
    save_path = os.path.join(get_project_dir(), 'models')

    epoch = 40
    loss_fn = nn.NLLLoss()

    for i in range(epoch):
        running_train_loss = 0
        running_val_loss = 0
        model.train()
        global_vars.isTrain = True
        
        for images, labels in trainloader:        
            optimizer.zero_grad()
            output = model(images.cuda())
            loss = loss_fn(output, labels.cuda())
            loss.backward()
            running_train_loss += loss.item()
            optimizer.step()
        
        model.eval()
        global_vars.isTrain = False
        with torch.no_grad():
            for images, labels in valloader:
                output = model(images.cuda())
                loss = loss_fn(output, labels.cuda())
                running_val_loss += loss.item()
            
            if i % 5 == 0:
                print('saving model...')
                torch.save({
                    'epoch': i,
                    'state_dict': model.state_dict(),
                    'validation loss' : running_val_loss,
                    'train loss' : running_train_loss}, os.path.join(save_path, 'model_' + str(i // 5)  + '.pth'))
            
        print('train_loss : ', running_train_loss / len(trainloader))
        print('val_loss : ', running_val_loss / len(valloader))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train script')
    parser.add_argument('--lr', default=0.03, nargs=1, help='learning rate')
    args = parser.parse_args()
    train(args.lr)


