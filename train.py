import argparse, os, sys, time, gc, datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import datetime

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Implementation of DiffMVS and CasDiffMVS')
parser.add_argument('--mode', help='train or test')
parser.add_argument('--device', default='cuda', help='select model')

# --------------Parameters for dataset----------------------
parser.add_argument('--dataset', default='dtu', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')
parser.add_argument('--trainviews', type=int, default=3,  help='trainviews')
parser.add_argument('--testviews', type=int, default=3,  help='testviews')

# --------------Parameters for training---------------------
parser.add_argument('--epochs', type=int, default=48, 
    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_sche', default='mslr', help='learning rate schedule')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2",
    help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument("--train_epochs", type=int, default=-1,
    help="number of epochs")
parser.add_argument('--batch_size', type=int, default=4,
    help='train batch size')
parser.add_argument('--seed', type=int, default=123, metavar='S',
    help='random seed')

parser.add_argument('--loadckpt', default=None,
    help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug/refine',
    help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true',
    help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=20,
    help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1,
    help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')

# --------------Parameters for model------------------------
parser.add_argument('--numdepth_initial', type=int, default=48,
    help='number of depth samples in deth initialization')
parser.add_argument('--numdepth', type=int, default=384,
    help='1.0/numdepth is the minimum sampling interval')
parser.add_argument('--ddim_eta', nargs="+", type=float, default=[0.01,0.01,0.01],
    help='eta for ddim')
parser.add_argument('--scale', nargs="+", type=float, default=[0.01,0.01,0.01],
    help='scale of noise')
parser.add_argument('--timesteps', nargs="+", type=int, default=[1000,1000,1000],
    help='timesteps')
parser.add_argument('--sampling_timesteps', nargs="+", type=int, default=[1,1,1],
    help='sampling_timesteps')
parser.add_argument('--hidden_dim', nargs="+", type=int, default=[0,32,32],
    help='timesteps')
parser.add_argument('--context_dim', nargs="+", type=int, default=[32,32,16], 
    help='timesteps')
parser.add_argument('--interval_scale', type=float, default=1.06,
    help='the number of depth values')
parser.add_argument('--stage_iters', nargs="+", type=int, default=[3,3,3],
    help='stage_iters')
parser.add_argument('--cost_dim_stage', nargs="+", type=int, default=[4,4,4],
    help='stage_iters')
parser.add_argument('--CostNum', nargs="+", type=int, default=[0, 4, 4],
    help='number of new samples in each diffusion timestep')
parser.add_argument('--unet_dim', nargs="+", type=int, default=[0,16,8],
    help='timesteps')
parser.add_argument('--conf_weight', type=float, default=1.0,
    help='weight for confidence learning')
parser.add_argument('--min_radius', type=float, default=0.2,
    help='min scale factor for sampling')
parser.add_argument('--max_radius', type=float, default=2,
    help='max scale factor for sampling')

# main function

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, EvalImgLoader, lr_scheduler, start_epoch, args):
    logger = SummaryWriter(args.logdir)
    if args.train_epochs == -1:
        total_epochs = args.epochs
    else:
        total_epochs = args.train_epochs
    for epoch_idx in range(start_epoch, total_epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx
        
        # training
        print_fre = len(TrainImgLoader) // 10
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            do_summary_image = global_step % (50*args.summary_freq) == 0
            loss, scalar_outputs, image_outputs = train_sample(model, model_loss, optimizer, sample, args)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
            if do_summary_image:
                save_images(logger, 'train', image_outputs, global_step)

            if args.lr_sche=="onecycle":
                lr_scheduler.step()

            print(
                   "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, depth loss = {:.3f}, time = {:.3f}".format(
                       epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                       optimizer.param_groups[0]["lr"], loss,
                       scalar_outputs['depth_loss'],
                       time.time() - start_time))

            del scalar_outputs
        if args.lr_sche=="mslr":
            lr_scheduler.step()

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        # gc.collect()

        # testing
        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(TestImgLoader):

                start_time = time.time()
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                do_summary_image = global_step % (50*args.summary_freq) == 0
                loss, scalar_outputs_test, image_outputs = test_sample_depth(model, model_loss, sample, args)
                scalar_outputs_test['time'] = time.time() - start_time
                if do_summary:
                    save_scalars(logger, 'test', scalar_outputs_test, global_step)
                if do_summary_image:
                    save_images(logger, 'test', image_outputs, global_step)
                avg_test_scalars.update(scalar_outputs_test)

            save_scalars(logger, 'full_test', avg_test_scalars.mean(), global_step)
            print("final", avg_test_scalars.mean())


def test(model, model_loss, TestImgLoader, args):
    avg_test_scalars = DictAverageMeter()
    i = 0
    print(len(TestImgLoader))
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
        scalar_outputs['time'] = time.time() - start_time
        avg_test_scalars.update(scalar_outputs)

        del scalar_outputs, image_outputs
    print("final", avg_test_scalars.mean())


def train_sample(model, model_loss, optimizer, sample, args):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]
    depth_values = sample_cuda["depth_values"]

    depth_gt = depth_gt_ms
    mask = mask_ms
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"],
                    sample_cuda["depth_values"], depth_gt_ms)
    outputs_depth = outputs["depth"]
    outputs_conf = outputs["conf"]

    loss, depth_loss_dict = model_loss(args, outputs_depth, outputs_conf, 
                                       depth_gt_ms, mask_ms, depth_values, 
                                       loss_rate=0.9, iters=args.stage_iters)

    if args.stage_iters[2]==0:
        iters = args.stage_iters[0]+args.stage_iters[1]+2
    else:
        iters = args.stage_iters[0]+args.stage_iters[1]+args.stage_iters[2]+3
    depth_est = outputs_depth[-1]
    depth_initial = outputs_depth[0]

    depth_loss = depth_loss_dict["l{}".format(iters-1)]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()

    scalar_outputs = {
        "loss": loss,
        "depth_loss": depth_loss,
        "init_abs_depth_error": AbsDepthError_metrics(depth_initial, depth_gt["stage1"],
                                                      mask["stage1"] > 0.5),
        "final_depth_error": AbsDepthError_metrics(depth_est, depth_gt["stage4"],
                                                   mask["stage4"] > 0.5),
    }

    for i in range(iters):
        scalar_outputs["l{}".format(i)] = depth_loss_dict["l{}".format(i)]
    image_outputs = {
        "depth_est": depth_est * mask["stage4"],
        "confidence": outputs["conf"][-1],
        "depth_est_nomask": depth_est,
        "depth_gt": sample["depth"]["stage1"],
        #  "ref_img": sample["imgs"][0][0]["stage1"],
        "errormap": (depth_est - depth_gt["stage4"]).abs() * mask["stage4"],
    }

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


@make_nograd_func
def test_sample_depth(model, model_loss, sample, args):
    model_eval = model
    model_eval.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]
    depth_values = sample_cuda["depth_values"]

    depth_gt = depth_gt_ms
    mask = mask_ms

    outputs = model_eval(
        sample_cuda["imgs"],
        sample_cuda["proj_matrices"],
        sample_cuda["depth_values"]
    )
    outputs_depth = outputs["depth"]
    outputs_conf = outputs["conf"]

    loss, depth_loss_dict = model_loss(
        args, outputs_depth, outputs_conf,
        depth_gt_ms, mask_ms, depth_values,
        loss_rate=0.9, iters=args.stage_iters
    )

    depth_est = outputs_depth[-1]
    depth_initial = outputs_depth[0]

    if args.stage_iters[2]==0:
        iters = args.stage_iters[0]+args.stage_iters[1]+2
    else:
        iters = args.stage_iters[0]+args.stage_iters[1]+args.stage_iters[2]+3
    depth_loss = depth_loss_dict["l{}".format(iters-1)]

    scalar_outputs = {
        "loss": loss,
        "depth_loss": depth_loss,
        "init_abs_depth_error": AbsDepthError_metrics(depth_initial, depth_gt["stage1"],
                                                      mask["stage1"] > 0.5),
        "final_depth_error": AbsDepthError_metrics(depth_est, depth_gt["stage4"],
                                                   mask["stage4"] > 0.5),

    }
    for i in range(iters):
        scalar_outputs["l{}".format(i)] = depth_loss_dict["l{}".format(i)]

    image_outputs = {
        "depth_est": depth_est * mask["stage4"],
        "depth_est_nomask": depth_est,
        "confidence": outputs["conf"][-1],
        "depth_gt": sample["depth"]["stage4"],
        #  "ref_img": sample["imgs"][0][0]["stage1"],
        "errormap": (depth_est - depth_gt["stage4"]).abs() * mask["stage4"]
    }

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)

if __name__ == '__main__':
    # parse arguments and check
    args = parser.parse_args()

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None
    if args.testpath is None:
        args.testpath = args.trainpath

    set_random_seed(args.seed)
    device = torch.device(args.device)

    if args.mode == "train":
        if not os.path.isdir(args.logdir):
            os.makedirs(args.logdir)
        current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        print("current time", current_time_str)
        print("creating new summary file")
        logger = SummaryWriter(args.logdir)
    print("argv:", sys.argv[1:])
    print_args(args)

    # model, optimizer
    model = CasDiffMVS(args)
    model.to(device)
    model_loss = compute_inverse_loss

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        eps=1e-8
    )

    # load parameters
    start_epoch = 0
    if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'], strict=False)
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])

    print("start at epoch {}".format(start_epoch))
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if torch.cuda.is_available():
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train",
                               args.trainviews, args.numdepth)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test",
                              args.testviews, args.numdepth)
    TrainImgLoader = DataLoader(train_dataset, args.batch_size,
                                shuffle=True, num_workers=8, drop_last=True)
    TestImgLoader = DataLoader(test_dataset, args.batch_size,
                                shuffle=False, num_workers=8, drop_last=False)


    EvalImgLoader = None

    if args.lr_sche=="mslr":
        milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
        lr_gamma = 1 / float(args.lrepochs.split(':')[1])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                        last_epoch=start_epoch - 1)
    elif args.lr_sche=="onecycle":
        last_epoch = len(TrainImgLoader) * start_epoch - 1 if start_epoch > 0 else -1
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, len(TrainImgLoader) * args.epochs + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear',
        last_epoch=last_epoch)
    else:
        raise NotImplementedError

    if args.mode == "train":
        train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, EvalImgLoader, lr_scheduler, start_epoch, args)
    elif args.mode == "test":
        test(model, model_loss, TestImgLoader, args)
    else:
        raise NotImplementedError
