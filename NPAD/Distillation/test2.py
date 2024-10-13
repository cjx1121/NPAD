from models.selector_test import *
from utils.util import *
from data_loader import get_test_loader, get_backdoor_loader
from config1 import get_arguments
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
# import models
import poison_cifar as poison
# from models import *
from models.wresnet_test import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output[3], labels).item()
            pred = output[3].data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

# def test(test_clean_loader, test_bad_loader, nets, criterions):
#     test_process = []
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#
#     snet = nets['snet']
#     criterionCls = criterions['criterionCls']
#     snet.eval()
#
#     for idx, (img, target) in enumerate(test_clean_loader, start=1):
#         img = img.cuda()
#         target = target.cuda()
#
#         with torch.no_grad():
#             _, _, _, output_s = snet(img)
#
#         prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
#         top1.update(prec1.item(), img.size(0))
#         top5.update(prec5.item(), img.size(0))
#
#     acc_clean = [top1.avg, top5.avg]
#
#     cls_losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#
#     for idx, (img, target) in enumerate(test_bad_loader, start=1):
#         img = img.cuda()
#         target = target.cuda()
#
#         with torch.no_grad():
#             _, _, _, output_s = snet(img)
#             cls_loss = criterionCls(output_s, target)
#
#         prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
#         cls_losses.update(cls_loss.item(), img.size(0))
#         top1.update(prec1.item(), img.size(0))
#         top5.update(prec5.item(), img.size(0))
#
#     acc_bd = [top1.avg, top5.avg, cls_losses.avg]
#
#     print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
#     print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))
#
#     return acc_clean, acc_bd


def main():
    # Prepare arguments
    device = 'cuda'
    opt = get_arguments().parse_args()
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])
    trigger_info = ''
    if trigger_info:
        trigger_info = torch.load(trigger_info, map_location=device)
    else:
        if opt.poison_type == 'benign':
            trigger_info = None
        else:
            triggers = {'badnets': 'checkerboard_1corner',
                        'clean-label': 'checkerboard_4corner',
                        'blend': 'gaussian_noise'}
            trigger_type = triggers[opt.poison_type]
            pattern, mask = poison.generate_trigger(trigger_type=trigger_type)
            trigger_info = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                            'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.poison_target])}      # poison_target默认为0


    clean_test = datasets.CIFAR10(root=opt.data_dir, train=False, download=True, transform=transform_test)
    poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)
    test_bad_loader = DataLoader(poison_test, batch_size=64, num_workers=0)
    test_clean_loader = DataLoader(clean_test, batch_size=64, num_workers=0)
    # net = wideresnet16()
    # print(net)
    net = torch.load('./weight/test/model.pt')
    student = net.to(device)
    nets = {'snet': student}
    summary(student, (3, 32, 32))
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = nn.CrossEntropyLoss()
    criterions = {'criterionCls': criterionCls}
    print('----------- DATA Initialization --------------')
    loss, acc = test(nets['snet'], criterions['criterionCls'], test_clean_loader)
    print("loss:{}".format(loss))
    print("acc:{}".format(acc))

if (__name__ == '__main__'):
    main()
