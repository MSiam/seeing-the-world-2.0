import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import argparse
import torch.nn as nn
import tqdm
import torch.optim as optim

def train(args):
    total_classes = 64
    val_every_nepochs = 1
    train_input_folder = args.data_dir + 'train/'
    validate_input_folder = args.data_dir + 'validate/'

    # Training Data Loader
    train_transform = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomVerticalFlip(),
       transforms.RandomAffine(scale=(0.9, 1.25), shear=0.2, degrees=0),
       transforms.Resize((224,224)),
       transforms.ToTensor(),
       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_dataset = torchvision.datasets.ImageFolder(train_input_folder,
                                                     transform=train_transform)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers)

    # Validation Data Loader
    val_transform = transforms.Compose([
       transforms.Resize((224,224)),
       transforms.ToTensor(),
       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_dataset = torchvision.datasets.ImageFolder(validate_input_folder,
                                                     transform=val_transform)
    val_loader = data.DataLoader(val_dataset, batch_size=1,
                                   shuffle=False, num_workers=args.num_workers)

    # Create Model
    model = torchvision.models.vgg19(pretrained=True)
    model.classifier[6] = nn.Linear(4096, total_classes, bias=True)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()

    # Create Optimizer and freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training/Val Loop
    tqdm_train = tqdm.tqdm(train_loader)
    tqdm_val = tqdm.tqdm(val_loader)
    model.train()
    for epoch in range(args.nepochs):
        for image, label in tqdm_train:
            image = image.cuda()
            label = label.cuda()

            optimizer.zero_grad()
            pred = model(image)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

        if epoch % val_every_nepochs == 0:
            model.eval()
            with torch.no_grad():
                for image, label in tqdm_val:
                    pred = model(image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str)
    parser.add_argument('-batch_size', type=int, default=30)
    parser.add_argument('-nepochs', type=int, default=80)
    parser.add_argument('-num_workers', type=int, default=1)
    args = parser.parse_args()
    train(args)

