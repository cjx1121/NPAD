from torch import nn
from tensorboardX import SummaryWriter
from models.selector_test import *
from utils.util import *
from data_loader import get_train_loader, get_test_loader
from at import AT
from config1 import get_arguments
import models
import matplotlib.pyplot as plt

def train_step(opt, train_loader, nets, optimizer, criterions, epoch):
    at_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionAT = criterions['criterionAT']

    snet.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        activation1_s, activation2_s, activation3_s, output_s = snet(img)
        activation1_t, activation2_t, activation3_t, _ = tnet(img)

        cls_loss = criterionCls(output_s, target)
        at3_loss = criterionAT(activation3_s, activation3_t.detach())
        at2_loss = criterionAT(activation2_s, activation2_t.detach())
        at1_loss = criterionAT(activation1_s, activation1_t.detach())
        at_loss = at1_loss + at2_loss + at3_loss + cls_loss

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        at_losses.update(at_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        at_loss.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:  # 默认为50
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'AT_loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=at_losses, top1=top1, top5=top5))


def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):
    test_process = []
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionAT = criterions['criterionAT']

    snet.eval()

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            _, _, _, output_s = snet(img)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))  # 更新准确率
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg]   # 干净测试集的top1和top5的准确率

    cls_losses = AverageMeter()
    at_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        img = img.cuda()
        target = target.cuda()

        with torch.no_grad():
            activation1_s, activation2_s, activation3_s, output_s = snet(img)
            activation1_t, activation2_t, activation3_t, _ = tnet(img)

            at3_loss = criterionAT(activation3_s, activation3_t.detach()) * opt.beta3
            at2_loss = criterionAT(activation2_s, activation2_t.detach()) * opt.beta2
            at1_loss = criterionAT(activation1_s, activation1_t.detach()) * opt.beta1
            at_loss = at3_loss + at2_loss + at1_loss
            cls_loss = criterionCls(output_s, target)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        at_losses.update(at_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, cls_losses.avg, at_losses.avg]   # 中毒测试集的准确率

    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

    # save training progress
    log_root = opt.log_root + '/results.csv'
    test_process.append((epoch, acc_clean[0], acc_bd[0], acc_bd[2], acc_bd[3]))
    df = pd.DataFrame(test_process, columns=("epoch", "test_clean_acc", "test_bad_acc", "test_bad_cls_loss", "test_bad_at_loss"))
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean, acc_bd


def train(opt):
    # Load models
    print('----------- Network Initialization --------------')
    # teacher = select_model(dataset=opt.data_name,         # 默认为cifar10
    #                        model_name=opt.t_name,         # 默认为WRN-16-1
    #                        pretrained=True,
    #                        pretrained_models_path=opt.t_model,  # 默认为./weight/t_net/WRN-16-1-T-model_best.pth.tar
    #                        n_classes=opt.num_class).to(opt.device)    # 默认为10
    arch = 'wideresnet16'
    device = opt.device
    # net = getattr(models, arch)(num_classes=10)
    # net.load_state_dict(torch.load(opt.t_model, map_location=device))
    teacher = torch.load("/home/cjx/python_project/ANP_backdoor/save2/model_last0.45.pt")
    teacher = teacher.to(device)

    print('finished teacher model init...')

    # student = select_model(dataset=opt.data_name,     # 默认为cifar10
    #                        model_name=opt.s_name,     # 默认为WRN-16-1
    #                        pretrained=True,
    #                        pretrained_models_path=opt.s_model,  # 默认为./weight/s_net/WRN-16-1-S-model_best.pth.tar
    #                        n_classes=opt.num_class).to(opt.device)  # 默认为10
    # net = getattr(models, arch)(num_classes=10)
    # print(torch.load(opt.s_model, map_location=device).keys())
    # net.load_state_dict(torch.load(opt.s_model, map_location=device)['state_dict'])

    student = torch.load("/home/cjx/python_project/ANP_backdoor/save2/model.pt")
    student = student.to(device)
    print('finished student model init...')
    teacher.eval()

    nets = {'snet': student, 'tnet': teacher}

    for param in teacher.parameters():
        param.requires_grad = False

    # initialize optimizer
    optimizer = torch.optim.SGD(student.parameters(),
                                lr=opt.lr,      # 默认为0.1
                                momentum=opt.momentum,  # 默认为0.9
                                weight_decay=opt.weight_decay, # 默认为1e-4
                                nesterov=True)

    # define loss functions
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
        criterionAT = AT(opt.p)       # 默认为2
    else:
        criterionCls = nn.CrossEntropyLoss()
        criterionAT = AT(opt.p)

    print('----------- DATA Initialization --------------')
    train_loader = get_train_loader(opt)
    test_clean_loader, test_bad_loader = get_test_loader(opt)  # 原文是5%的干净训练数据

    print('----------- Train Initialization --------------')
    for epoch in range(0, opt.epochs):  # 默认20轮

        adjust_learning_rate(optimizer, epoch, opt.lr)  # 20轮的时候lr=1e-4

        # train every epoch
        criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}

        if epoch == 0:
            # before training test firstly
            test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch)  # 得到CLA和ASR

        # 教师网络教导学生网络过程
        train_step(opt, train_loader, nets, optimizer, criterions, epoch+1)  # 为什么没有输出不理解，因为不足50

        # evaluate on testing set
        print('testing the models......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch+1)
        with open('acc1.txt', 'a+') as f:
            f.write(str(acc_clean[0]) + '\n')
        with open('loss1.txt', 'a+') as f:
            f.write(str(acc_bad[2]) + '\n')
        with open('asr1.txt', 'a+') as f:
            f.write(str(acc_bad[0]) + '\n')
        with open('atloss1.txt', 'a+') as f:
            f.write(str(acc_bad[3]) + '\n')

        # remember best precision and save checkpoint
        # save_root = opt.checkpoint_root + '/' + opt.s_name
        if opt.save:
            is_best = acc_clean[0] > opt.threshold_clean  # CLA默认为90
            opt.threshold_clean = min(acc_bad[0], opt.threshold_clean)

            best_clean_acc = acc_clean[0]
            best_bad_acc = acc_bad[0]

            save_checkpoint({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_clean_acc': best_clean_acc,
                'best_bad_acc': best_bad_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, opt.checkpoint_root, opt.s_name)


def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    train(opt)

if (__name__ == '__main__'):
    main()
