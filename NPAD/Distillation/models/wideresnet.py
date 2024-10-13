from models.selector_test import *
from utils.util import *
from data_loader import get_test_loader, get_backdoor_loader
from config1 import get_arguments
from torchsummary import summary
import models
from models import *
from models.wresnet_test import *

def test(opt, test_clean_loader, test_bad_loader, nets, criterions):
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


def main():
    # Prepare arguments
    device = 'cuda'
    opt = get_arguments().parse_args()
    test_clean_loader, test_bad_loader = get_test_loader(opt)
    net = wideresnet16()
    print(net)
    net = torch.load('./weight/test/t_net/model_last.pt')
    student = net.to(device)
    nets = {'snet': student}
    # # summary(student, (3, 32, 32))
    # if opt.cuda:
    #     criterionCls = nn.CrossEntropyLoss().cuda()
    # else:
    #     criterionCls = nn.CrossEntropyLoss()
    # criterions = {'criterionCls': criterionCls}
    # print('----------- DATA Initialization --------------')
    # acc_clean, acc_bd = test(opt, test_clean_loader, test_bad_loader, nets, criterions)
    # print("acc_clean:{}".format(acc_clean))
    # print("acc_bd:{}".format(acc_bd))

if (__name__ == '__main__'):
    main()