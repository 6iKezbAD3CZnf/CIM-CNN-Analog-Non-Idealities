import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models import MLP, LeNet, MisMatchedLeNet

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def train(args, train_loader, model, optimizer, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f"Training loss at batch {batch_idx}: {loss.item():.4f}", end='\r')

def test(args, test_loader, model, device, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f"Epoch {epoch+1}: test loss {test_loss:.4f}, accuracy {correct / len(test_loader.dataset) * 100:.2f}.")

def main():
    use_gpu = not args.no_gpu and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_gpu else "cpu")
    torch.manual_seed(args.seed)

    gpu_args = {'num_workers': 12, 'pin_memory': True} if use_gpu else {}
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(args.data_dir, train=True, download=True,
                                   transform=mnist_transform),
        batch_size=args.batch_size, shuffle=True, **gpu_args)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(args.data_dir, train=False, transform=mnist_transform),
        batch_size=args.test_batch_size, shuffle=True, **gpu_args)

    model = MisMatchedLeNet(
            linear_variation_range=args.linear_variation_range,
            relu_offset_range=args.relu_offset_range
            ).to(device)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model.update_variation(
                    linear_variation_range=args.linear_variation_range,
                    relu_offset_range=args.relu_offset_range
                    )
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        test(args, test_loader, model, device, epoch=0)
        return

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.start_epoch, args.epochs):
        train(args, train_loader, model, optimizer, device, epoch)
        test(args, test_loader, model, device, epoch)

    save_checkpoint({
        'epoch' : args.epochs,
        'state_dict': model.state_dict(),
    }, filename=os.path.join(args.save_dir, 'model.th'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='data directory (default: ./data)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='hidden size of the fully connected layer (default: 256)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='disables GPU training')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device to use (default: 0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status (default: 100)')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='save_temp', type=str)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--linear-variation-range', type=float, default=0.1,
                        help='the range of linear variation (default: 0.1)')
    parser.add_argument('--relu-offset-range', type=float, default=1.0,
                        help='the range of relu offset (default: 1.0)')

    args = parser.parse_args()

    main()
