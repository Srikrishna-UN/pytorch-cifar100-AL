
from torch.utils.data import DataLoader
from utils import *
from methods import *
import argparse
from conf import settings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-m', type=int, default=1, help='method used for sampling')
    args = parser.parse_args()

    model = get_network(args)

    cifar100_training_data, rest = get_training_data(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
    )

    cifar100_test_data = get_test_data(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
    )

    cifar100_training_loader = DataLoader(cifar100_training_data,shuffle=True,num_workers=4,batch_size=128)
    cifar100_test_loader = DataLoader(cifar100_test_data,shuffle=True,num_workers=4,batch_size=128)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)

    train_model(model,cifar100_training_loader,loss_function,optimizer,train_scheduler,50)
    ini_acc = test_model(model,cifar100_test_loader)
    print(f"Initial acc on 10k:{ini_acc}")
    train_until_empty(model, cifar100_training_data, ini_acc, rest, cifar100_test_data, max_iterations=15, batch_size=128,
                      learning_rate=0.01, method=args.m)

