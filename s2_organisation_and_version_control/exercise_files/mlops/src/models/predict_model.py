import sys
from helper import get_source_dir, get_project_dir
sys.path.append(get_source_dir())
import os
import global_vars
import argparse
import torch
from features.build_features import mnist
from model import MyAwesomeModel

def evaluate(model_checkpoint : str):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    
    _, test_dataset = mnist()
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    model.eval()
    global_vars.isTrain = False
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            output = model(images)
            prediction = torch.argmax(output, axis=-1)
            correct += torch.sum(prediction == labels).item()
            total += len(labels)
        
        print('accuracy : ', correct / total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train script')
    parser.add_argument('--checkpoint', default=os.path.join(get_project_dir(), 'models/model_7.pth'), nargs=1, help='checkpoint path .pth file')
    args = parser.parse_args()
    evaluate(args.checkpoint)

