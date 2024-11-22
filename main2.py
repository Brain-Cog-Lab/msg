import sys
sys.path.append("../..")
import matplotlib
matplotlib.use('agg')

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import *
from surrogate import *
from neuron import *
from snn_model import SpikeNet
from model import *
from utils import result2csv, seed_all, setup_default_logging, accuracy, AverageMeter

import matplotlib.pyplot as plt
import datetime, time, argparse, logging, math, os
import wandb

try:
    from apex import amp
except:
    print('no apex pakage installed')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remark', type=str, default='')
    parser.add_argument('--model', type=str, default='vggnet', help="'cifarconvnet', 'vgg16', 'resnet18', 'resnet34', 'resnet50'")
    parser.add_argument('--dataset', type=str, default='dvscifar10',
                        choices=['mnist', 'fashionmnist', 'cifar10', 'cifar100', 'imagenet', 'dvsgesture', 'dvscifar10', 'ncaltech101', 'ncars', 'nmnist'])
    parser.add_argument('--data_path', type=str, default='/data/datasets', help='/Users/lee/data/datasets')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_every', type=int, default=1)

    # Optimizer and lr_scheduler
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--optim', type=str, default='adamW', choices=['adamW', 'adam', 'sgd'])
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'mse'])
    parser.add_argument('--schedu', type=str, default='cosin', choices=['step', 'mstep', 'cosin'])
    parser.add_argument('--step_size', type=int, default=50, help='parameter for StepLR')
    parser.add_argument('--milestones', type=list, default=[150, 250])
    parser.add_argument('--lr_gamma', type=float, default=0.05)
    parser.add_argument('--warmup', type=float, default=5)
    parser.add_argument('--warmup_lr_init', type=float, default=1e-6)
    parser.add_argument('--init_lr', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # Path
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--saved_dir', type=str, default='/data/ly/msg_ijcai')
    parser.add_argument('--saved_csv', type=str, default='./results_ijcai.csv')
    parser.add_argument('--save_log', type=bool, default=False)
    # parser.add_argument('--save_log', action='store_false')

    # Device
    parser.add_argument('--device', type=str, default='6')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--apex', type=bool, default=False)
    parser.add_argument('--amp', type=bool, default=False)
    parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN.')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dali', type=bool, default=False)
    parser.add_argument('--channel_last', type=bool, default=False)

    # SNN
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--encode_type', type=str, default='direct')

    ## neuron
    parser.add_argument('--neuron', type=str, default='LIF')
    parser.add_argument('--tau', type=float, default=2.)
    parser.add_argument('--threshold', type=float, default=.5)

    # surrogate
    parser.add_argument('--act_func', type=str, default='PiecewiseLinearGrad')
    parser.add_argument('--alpha', type=float, default=2.)
    parser.add_argument('--alpha_grad', type=bool, default=False)
    parser.add_argument('--sub_func', type=str, default='StepGrad')
    parser.add_argument('--sub_prob', type=float, default=0.0)
    parser.add_argument('--mix_mode', type=str, default='rand')

    parser.add_argument('--name', default='', type=str)
    # parser.add_argument('--attout', action='store_true')
    parser.add_argument('--attout', type=bool, default=True)
    parser.add_argument('--att_alpha', type=float, default=0.3)
    parser.add_argument('--att_startepoch', type=int, default=0)

    return parser.parse_args()


def count_output(net, y):
    # outmax = torch.cumsum(net.outputs, dim=0).argmax(dim=2)
    outmax = net.outputs.argmax(dim=2)
    return (outmax.detach() == y).float().mean(1).cpu()


def train_net(net, train_iter, test_iter, optimizer, scheduler, criterion, device, args=None):
    best = 0
    net = net.to(device)
    class_num = args.num_classes
    plt.figure()


    if args.amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        if args.attout:
            if epoch <= args.att_startepoch:
                net.att = torch.ones([args.T, 1, 1]).to(device)
            else:
                new = (time_tot.avg / time_tot.avg.mean(0)).reshape(args.T, 1, 1).to(device)

                net.att = net.att * (1 - args.att_alpha) + new * args.att_alpha
                print(time_tot.avg / time_tot.avg.mean(0))

                new = new.detach().cpu()
                dic = {"new1": new[0,0,0].item(),
                       "new2": new[1, 0, 0].item(),
                       "new3": new[2, 0, 0].item(),
                       "new4": new[3, 0, 0].item(),
                       "new5": new[4, 0, 0].item(),
                       "new6": new[5, 0, 0].item(),
                       "new7": new[6, 0, 0].item(),
                       "new8": new[7, 0, 0].item(),
                       "new9": new[8, 0, 0].item(),
                       "new10": new[9, 0, 0].item(),

                       "att1": net.att[0, 0, 0].item(),
                       "att2": net.att[1, 0, 0].item(),
                       "att3": net.att[2, 0, 0].item(),
                       "att4": net.att[3, 0, 0].item(),
                       "att5": net.att[4, 0, 0].item(),
                       "att6": net.att[5, 0, 0].item(),
                       "att7": net.att[6, 0, 0].item(),
                       "att8": net.att[7, 0, 0].item(),
                       "att9": net.att[8, 0, 0].item(),
                       "att10": net.att[9, 0, 0].item(),
                       }
                result2csv(os.path.join('./', 'att_600_no.csv'), dic)

            print('epoch: %s, factor:'% epoch, net.att.squeeze(1).squeeze(1))

        loss_tot = AverageMeter()
        acc_tot = AverageMeter()
        time_tot = AverageMeter()
        start = time.time()
        net.train()

        for ind, data in enumerate(train_iter):
            if args.local_rank == int(args.device):
                tim = int(time.time()-start)
                pert = tim/(ind+1)
                eta = pert * (len(train_iter)-ind)
                print("Training iter:", str(ind)+'/'+str(len(train_iter)),
                      '['+ '%02d:%02d' % (tim//60, tim%60)+'<'+
                      '%02d:%02d,' % (eta // 60, eta % 60),
                      '%.2f s/it]' % pert,
                      end='\r'
                      )
            if args.dali:
                X = data[0]['data'].to(device, non_blocking=True)
                y = data[0]['label'].squeeze(-1).long().to(device, non_blocking=True)
            else:
                X = data[0].to(device, non_blocking=True)
                y = data[1].to(device, non_blocking=True)

            output = net(X)
            label = F.one_hot(y, class_num).float() if isinstance(criterion, torch.nn.MSELoss) else y
            loss = criterion(output, label)

            optimizer.zero_grad()
            if args.apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()

            elif args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()
                optimizer.step()

            acc, = accuracy(output, y, topk=(1,))
            if args.attout: time_tot.update(count_output(net, y), y.shape[0])
            loss_tot.update(loss.item(), output.shape[0])
            acc_tot.update(acc, output.shape[0])

        scheduler.step()
        if args.local_rank == int(args.device):
            args.logger.info('-'*10+ 'Epoch:' + str(epoch + 1)+ '-'*10 + '\n' +
                '<Train>  acc:%.6f, loss:%.6f, lr:%.6f, time:%.1f s'
                  % (acc_tot.avg, loss_tot.avg, optimizer.param_groups[0]['lr'], time.time() - start))

        if args.saved_dir is not None:
            saved_dir = os.path.join(args.saved_dir, 'checkpoints.pth')

        if epoch % args.eval_every == 0:
            test_acc, test_loss = evaluate_net(test_iter, net, criterion, device, args)

            if test_acc > best:
                best = test_acc

                if args.save_log:
                    torch.save(net.state_dict(), saved_dir)
                    args.logger.info("saved model on"+ saved_dir+"-"+ str(best))

        args.logger.info("<Best> acc:%.6f \n" % best)

        # wandb.log({"epoch": epoch,
        #            "train_loss": loss_tot.avg,
        #            "test_loss": test_loss,
        #            "train_acc": acc_tot.avg.item(),
        #            "test_acc": test_acc.item(),
        #            "best": best,
        #            })

        if args.save_log:
            dic = {"epoch": epoch,
                   "train_loss": loss_tot.avg,
                   "test_loss": test_loss,
                   "train_acc" : acc_tot.avg.item(),
                   "test_acc": test_acc.item(),
                   }
            result2csv(os.path.join(args.saved_dir, 'result.csv'), dic)


    args.logger.info("Best test acc: %.6f" % best)

    args.acc = best.detach().cpu().item()
    result2csv(args.saved_csv, args)
    print("Write results to csv file.")


def evaluate_net(data_iter, net, criterion, device, args=None):
    class_num = args.num_classes
    net = net.to(device)
    loss_tot = AverageMeter()
    acc_tot = AverageMeter()
    acc_tot_t = AverageMeter()
    with torch.no_grad():
        start = time.time()
        for ind, data in enumerate(data_iter):
            if args is None:
                print("Testing", str(ind) + '/' + str(len(data_iter)), end='\r')
            elif args.local_rank == int(args.device):
                print("Testing", str(ind) + '/' + str(len(data_iter)), end='\r')

            X = data[0].to(device, non_blocking=True)
            y = data[1].to(device, non_blocking=True)

            net.eval()
            output = net(X.to(device)).detach()

            a = net.att
            net.att = torch.ones_like(net.att).to(device) / args.T
            output1 = net(X.to(device)).detach()
            net.att = a
            net.train()
            label = F.one_hot(y, class_num).float() if isinstance(criterion, torch.nn.MSELoss) else y
            loss = criterion(output, label)

            acc, = accuracy(output, y.to(device), topk=(1,))
            acc1, = accuracy(output1, y.to(device), topk=(1,))
            acc_tot.update(acc, output.shape[0])
            acc_tot_t.update(acc1, output1.shape[0])
            loss_tot.update(loss.item(), output.shape[0])

    if args is None:
        print('<Test>   acc:%.6f, time:%.1f s' % (acc_tot.avg, time.time() - start))
    elif args.local_rank == int(args.device):
        args.logger.info('<Test>   acc:%.6f, acc_wotwo:%.6f, time:%.1f s' % (acc_tot.avg, acc_tot_t.avg, time.time() - start))
    return acc_tot.avg, loss_tot.avg


if __name__ == "__main__":
    args = parse_args()
    seed_all(args.seed)
    args.logger = logging.getLogger('train')

    # wandb.init(project='MSG', resume="allow")
    # wandb.config.update(vars(args))

    now = (datetime.datetime.now()+datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")
    args.time = now
    exp_name = '-'.join([
        now,
        args.model,
        args.dataset,
        str(args.T),
        args.act_func,
        args.sub_func,
        str(args.sub_prob),
        args.mix_mode,
        str(args.seed)])

    args.saved_dir = os.path.join(args.saved_dir, exp_name)
    if args.save_log: os.makedirs(args.saved_dir, exist_ok=True)
    setup_default_logging(log_path=os.path.join(args.saved_dir, 'log.txt') if args.save_log else None)

    CKPT_DIR = os.path.join(args.saved_dir, exp_name)

    world_size = 1
    if args.distributed:
        world_size = torch.distributed.get_world_size()
    else:
        os.environ["LOCAL_RANK"] = args.device

    local_rank = int(os.environ["LOCAL_RANK"])
    args.local_rank = local_rank

    if args.local_rank == int(args.device):
        args_dict = "Namespace:\n"
        for eachArg, value in args.__dict__.items():
            args_dict += eachArg + ' : ' + str(value) + '\n'
        args.logger.info(args_dict)

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", local_rank)

        if args.distributed:
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(backend='nccl')
    else:
        device = torch.device("cpu")

    if args.dataset in ['mnist', 'fashionmnist']:
        train_loader = eval('get_%s_data' % args.dataset)(
            args.data_path, args.batch_size, train=True, num_workers=args.num_workers)
        test_loader = eval('get_%s_data' % args.dataset)(
            args.data_path, args.batch_size, train=False, num_workers=args.num_workers)
        in_channels = 1

    elif args.dataset in ['cifar10', 'cifar100', 'imagenet']:
        train_loader = eval('get_%s_data' % args.dataset)(
            args.data_path, args.batch_size, train=True,
            num_workers=args.num_workers, distributed=args.distributed)
        test_loader = eval('get_%s_data' % args.dataset)(
            args.data_path, args.batch_size, train=False,
            num_workers=args.num_workers, distributed=args.distributed)
        in_channels = 3
    elif args.dataset in ['dvsgesture', 'dvscifar10', 'ncaltech101', 'ncars', 'nmnist']:
        train_loader = eval('get_%s_data' % args.dataset)(
            args.data_path, args.batch_size, T=args.T, train=True, num_workers=args.num_workers)
        test_loader = eval('get_%s_data' % args.dataset)(
            args.data_path, args.batch_size, T=args.T, train=False, num_workers=args.num_workers)
        in_channels = 2
    else:
        raise NotImplementedError("Can't find the dataset loader")

    args.num_classes = num_classes = cls_num_classes[args.dataset]

    if args.model == 'mnistconvnet':
        model = MNISTConvNet()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'cifarconvnet':
        model = CIFARConvNet()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'dvsconvnet':
        model = DVSConvNet()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'vggnet':
        model = VGGNet()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'vgg16':
        model = VGG16('avg')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'resnet18':
        model = ResNet18(pool=args.pool)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == 'resnet18v2':
        model = ResNet18v2()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise NotImplementedError

    # if args.resume != '':
    #     args.logger.info('keep training model: %s' % args.resume)
    #     model.load_state_dict(torch.load(args.resume, map_location=device))

    if args.act_func != 'ann':
        surrofunction = eval(args.act_func)(
            alpha=args.alpha,
            requires_grad=args.alpha_grad,
            sub_func=args.sub_func,
            sub_prob=args.sub_prob,
            mix_mode=args.mix_mode)

        neuronmodel = eval(args.neuron)(
            act_func=surrofunction,
            threshold=args.threshold,
            tau=args.tau
        )

        model = SpikeNet(model,
                         T=args.T,
                         encode_type=args.encode_type,
                         neuron=neuronmodel,
                         in_channels=in_channels
                         )

    if args.resume != '':
        args.logger.info('keep training model: %s' % args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=device))

    args.logger.info(model)

    args.init_lr = args.init_lr * args.batch_size * world_size / 1024.0
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    else:
        raise NotImplementedError

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.distributed:
        model = DDP(model,device_ids=[local_rank], output_device=local_rank)

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.loss == 'mse':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError

    if args.schedu == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.lr_gamma)
    elif args.schedu == 'mstep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.lr_gamma)
    elif args.schedu == 'cosin':
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0)
        warm_up_with_cosine_lr = lambda epoch: epoch / args.warmup * (1 - args.warmup_lr_init) + args.warmup_lr_init if epoch < args.warmup \
            else args.min_lr + 0.5 * (1.0 - args.min_lr) * (math.cos((epoch - args.warmup) / (args.epochs - args.warmup) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    else:
        raise NotImplementedError


    train_net(model,
              train_loader, test_loader,
              optimizer, scheduler, criterion,
              device,
              args)