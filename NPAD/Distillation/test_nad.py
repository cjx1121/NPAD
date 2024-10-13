from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from models.selector_test import *
from utils.util import *
from data_loader import get_test_loader, get_backdoor_loader
from config1 import get_arguments
from torchsummary import summary
# import models
# from models import *
from models.wresnet_test import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def test(model, criterion, data_loader):
#     model.eval()
#     total_correct = 0
#     total_loss = 0.0
#     with torch.no_grad():
#         for i, (images, labels) in enumerate(data_loader):
#             images, labels = images.to(device), labels.to(device)
#             output = model(images)
#             total_loss += criterion(output[3], labels).item()
#             pred = output[3].data.max(1)[1]
#             total_correct += pred.eq(labels.data.view_as(pred)).sum()
#     loss = total_loss / len(data_loader)
#     acc = float(total_correct) / len(data_loader.dataset)
#     return loss, acc

def test(test_clean_loader, test_bad_loader, nets, criterions):
    test_process = []
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    criterionCls = criterions['criterionCls']
    snet.eval()

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            _, _, _, output_s = snet(img)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg]

    cls_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            _, _, _, output_s = snet(img)
            cls_loss = criterionCls(output_s, target)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, cls_losses.avg]

    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

    return acc_clean, acc_bd

import torchvision.transforms as transforms

def main():
    # Prepare arguments
    device = 'cuda'
    opt = get_arguments().parse_args()
    # MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    # STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    # ])
    # clean_test = CIFAR10(root='data/CIFAR10', train=False, download=True, transform=transform_test)
    # clean_test_loader = DataLoader(clean_test, batch_size=64, num_workers=0)
    test_clean_loader, test_bad_loader = get_test_loader(opt)
    # net = wideresnet16()
    # print(net)
    # net = torch.load('./weight/test/model.pt')
    # net = wideresnet16()
    # temp = torch.load("./weight/test/WRN-16-1.tar")
    # print(temp.keys())
    # net.load_state_dict(torch.load("./weight/test/WRN-16-1.tar")['state_dict'])
    net = torch.load('/home/cjx/python_project/ANP_backdoor/save3/model.pt')
    # net=torch.load("/home/cjx/python_project/ANP_backdoor/save1/model_last0.05.pt")
    student = net.to(device)

    nets = {'snet': student}
    # summary(student, (3, 32, 32))
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = nn.CrossEntropyLoss()
    criterions = {'criterionCls': criterionCls}
    print('----------- DATA Initialization --------------')

    test(test_clean_loader, test_bad_loader, nets, criterions)
    # loss, acc = test(nets['snet'], criterions['criterionCls'], test_clean_loader)

    # print("loss:{}".format(loss))
    # print("acc:{}".format(acc))

if (__name__ == '__main__'):
    main()



# import torch
# from torch import nn
# import torchvision
# from torch.utils.data import DataLoader
#
# device = torch.device('cuda')
# loss_fn = nn.CrossEntropyLoss()
# if torch.cuda.is_available():
#     loss_fn = loss_fn.to(device)
#
# test_data = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor())
# test_dataloader = DataLoader(test_data, batch_size=64)
# test_data_size = len(test_data)
#
# net = torch.load('./weight/test/model.pt')
# net.eval()
# total_test_loss = 0
# total_accuracy = 0
# with torch.no_grad():
#     for data in test_dataloader:
#         images, targets = data
#         images = images.to(device)
#         targets = targets.to(device)
#         outputs = net(images)[3]
#         loss = loss_fn(outputs, targets)
#         # print(type(loss))
#         total_test_loss = total_test_loss + loss.item()
#         accuracy = (outputs.argmax(1) == targets).sum()
#         total_accuracy = total_accuracy + accuracy
#
# print("整体测试集上的loss:{}".format(total_test_loss))
# print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))