import argparse
import torchvision
from torch.utils.data import DataLoader
from TNCS.run import CSGAN_Trainer
import torchvision.transforms as transforms

# 随机仿射变换
# transform = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)

# set Training parameter
parser = argparse.ArgumentParser(description='Compressed sensing with NN')
# Set super parameters
parser.add_argument('--batchSize', type=int, default=8, help='Small batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='Test batch size')
parser.add_argument('--nEpochs', type=int, default=1000, help='Iterations')
parser.add_argument('--imageSize', type=int, default=64, metavar='N')
parser.add_argument('--sensing_rate', type=float, default=0.11, help='set sensing rate')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of the Net')
parser.add_argument('--trainPath', default='./dataset/train/')
parser.add_argument('--valPath', default='./dataset/test_images')
parser.add_argument('--lr', type=float, default=0.0004, help='Learning Rate. Default=0.001')
parser.add_argument('--seed', type=int, default=22, help='40% 34  10% 22  random seed to use.')
parser.add_argument('--last_lr', default=5e-5, type=float, help='initial learning rate')
parser.add_argument('--warm_epochs', default=3, type=int, help='number of epochs to warm up')
parser.add_argument('--cuda', action='store_true', default=True)
args = parser.parse_args()
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}


transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(args.imageSize),
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.RandomHorizontalFlip(p=0.2),
    torchvision.transforms.ToTensor(),
])

transforms_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
])


train_dataset = torchvision.datasets.ImageFolder(args.trainPath, transform=transforms)
val_dataset = torchvision.datasets.ImageFolder(args.valPath, transform=transforms_test)
train_loader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.testBatchSize, shuffle=False)



model = CSGAN_Trainer(args, train_loader, val_loader)

model.run()


