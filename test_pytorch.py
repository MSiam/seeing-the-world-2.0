import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import argparse
import torch.nn as nn
import tqdm
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from model import VanillaVGG19, CosSimVGG19

def test(args):
    total_classes = 64
    validate_input_folder = args.data_dir + 'validate/'

    # Validation Data Loader
    val_transform = transforms.Compose([
       transforms.Resize((224,224)),
       transforms.ToTensor(),
       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_dataset = torchvision.datasets.ImageFolder(validate_input_folder,
                                                     transform=val_transform)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)

    # Create Model
    if args.model_type == 'vanilla':
        model = VanillaVGG19(num_classes=total_classes)
    else:
        model = CosSimVGG19(num_classes=total_classes)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()

    model.load_state_dict(torch.load('ckpt.pth'))

    model.eval()
    with torch.no_grad():
        correct = 0.0
        metrics = {'macro': {'prec': 0.0, 'rec': 0.0, 'fscore': 0.0},
                   'micro': {'prec': 0.0, 'rec': 0.0, 'fscore': 0.0},
                   'weighted': {'prec': 0.0, 'rec': 0.0, 'fscore': 0.0}}

        tqdm_val = tqdm.tqdm(val_loader)
        for image, label in tqdm_val:
            image = image.cuda()
            pred = model(image)
            pred_labels = pred.detach().cpu().argmax(dim=1)
            for key in metrics.keys():
                prec, recall, fscore, _ = precision_recall_fscore_support(label, pred_labels,
                                                                          average=key)
                metrics[key]['prec'] += prec
                metrics[key]['rec'] += recall
                metrics[key]['fscore'] += fscore
            correct += (pred_labels == label).sum().item()
        print('Accuracy ', correct/(len(tqdm_val)*args.batch_size))
        for key in metrics.keys():
            print(' Metric %s: prec rec fscore'%key)
            out_str = ''
            for k2 in metrics[key].keys():
                metrics[key][k2] /= (len(tqdm_val))
                out_str += str(metrics[key][k2]) + ' '
            print(out_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str)
    parser.add_argument('-model_type', type=str, default='vanilla')
    parser.add_argument('-batch_size', type=int, default=30)
    parser.add_argument('-num_workers', type=int, default=1)
    args = parser.parse_args()
    test(args)

