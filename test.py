import argparse, os, time, sys, gc, cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import save_pfm, write_cam
from filter import *


os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Get depth maps and fuse them into a point cloud')
# --------------General arameters --------------------------
parser.add_argument('--method', type=str, default='casdiffmvs')
parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs_cas', help='output dir')
parser.add_argument('--seed', type=int, default=123, metavar='S', help='random seed')
parser.add_argument('--save_depth', action='store_true', help='save depth map')

# --------------Parameters for dataset----------------------
parser.add_argument('--dataset', default='general', help='select dataset')
parser.add_argument('--testpath', help='testing data dir for some scenes')
parser.add_argument('--testlist', help='testing scene list')
parser.add_argument('--num_view', type=int, default=5, help='num of view')
parser.add_argument('--max_h', type=int, default=4800, help='testing max h')
parser.add_argument('--max_w', type=int, default=6400, help='testing max w')

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
parser.add_argument('--min_radius', type=float, default=0.2,
    help='min scale factor for sampling')
parser.add_argument('--max_radius', type=float, default=2,
    help='max scale factor for sampling')

# --------------Parameters for post-processing--------------
parser.add_argument('--geo_mask_thres', type=int, default=2,
    help='depth should be consistent in at least N neighboring view')
parser.add_argument('--geo_pixel_thres', type=float, default=1,
    help='pixel error threshold for geometric consistency filtering')
parser.add_argument('--geo_depth_thres', type=float, default=0.01,
    help='depth error threshold for geometric consistency filtering')
parser.add_argument('--photo_thres', nargs="+", type=float, default=[0.3,0,0],
    help='confidence threshold for photometric consistency filtering')


# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

set_random_seed(args.seed)

def save_depth(testlist):
    for scene in testlist:
        avg_time = save_scene_depth([scene])
        print("avg_time", avg_time)

def save_scene_depth(testlist):
    # dataset, dataloader
    MVSDataset = find_dataset_def("mvs")
    test_dataset = MVSDataset(
        args.testpath, args.num_view, args.numdepth,
        dataset=args.dataset, scan=testlist,
        max_h=args.max_h, max_w=args.max_w
    )

    TestImgLoader = DataLoader(
        test_dataset, args.batch_size,
        shuffle=False, num_workers=2, drop_last=False
    )

    model = CasDiffMVS(args, test=True)
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=False)
    model.cuda()
    model.eval()
    
    time_sum = 0.0
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            depth_max = 1. / sample["depth_values"][:, 0]
            depth_min = 1. / sample["depth_values"][:, -1]

            sample_cuda = tocuda(sample)
            depth_max = tensor2numpy(depth_max)
            depth_min = tensor2numpy(depth_min)
            torch.cuda.synchronize()
            start_time = time.time()
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"],
                            sample_cuda["depth_values"])
            torch.cuda.synchronize()
            end_time = time.time()
            del sample_cuda

            outputs = tensor2numpy(outputs)
            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage4"].numpy()
            imgs = sample["imgs"][0].numpy()
            print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(TestImgLoader),
                                                      end_time - start_time, imgs[0].shape))
            time_sum = time_sum + end_time - start_time

            confs = outputs["photometric_confidence"]
            # # save depth maps and confidence maps
            if args.method == 'casdiffmvs':
                """CasDiffMVS"""
                for filename, cam, img, depth_est, depth_max_, depth_min_ in zip(
                    filenames, cams, imgs, outputs["depth"][-1], depth_max, depth_min
                ):
                    # img = img[0]  # ref view
                    print(img.shape)
                    cam = cam[0]  # ref cam

                    depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                    cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                    img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                    os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)

                    # save depth maps
                    save_pfm(depth_filename, depth_est)
                    # save cam, img
                    write_cam(cam_filename, cam, depth_max_, depth_min_)
                    img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_filename, img_bgr)
                    
                # save confidence maps
                for i in range(3):
                    cur_conf = confs[i].squeeze(0)
                    conf_name = 'conf{}'.format(i)
                    conf_filename = os.path.join(args.outdir, filename.format(conf_name, '.pfm'))
                    os.makedirs(conf_filename.rsplit('/', 1)[0], exist_ok=True)
                    save_pfm(conf_filename, cur_conf)
            else:
                """DiffMVS"""
                for filename, cam, img, depth_est,  depth_max_, depth_min_ in zip(
                    filenames, cams, imgs, outputs["depth"][-1], depth_max, depth_min
                ):
                    # img = img[0]  # ref view
                    cam = cam[0]  # ref cam

                    depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                    cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                    img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                    os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)

                    # save depth maps
                    save_pfm(depth_filename, depth_est)
                    # save cam, img
                    write_cam(cam_filename, cam, depth_max_, depth_min_)
                    img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_filename, img_bgr)
                
                # save confidence maps
                for i in range(2):
                    cur_conf = confs[i].squeeze(0)
                    conf_name = 'conf{}'.format(i)
                    conf_filename = os.path.join(args.outdir, filename.format(conf_name, '.pfm'))
                    os.makedirs(conf_filename.rsplit('/', 1)[0], exist_ok=True)
                    save_pfm(conf_filename, cur_conf)

    torch.cuda.empty_cache()
    gc.collect()
    avg_time = time_sum / len(TestImgLoader)
    return avg_time

if __name__ == '__main__':
    if args.dataset == 'dtu':
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
    elif args.dataset == 'tank':
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
    
        photo_thres_all = {
            "Family": [0.8, 0.8, 0.95],
            "Francis": [0.3, 0.6, 0.6],
            "Horse": [0.15, 0.4, 0.8],
            "Lighthouse": [0.3, 0.8, 0.9],
            "M60": [0.7, 0.8, 0.95],
            "Panther": [0.3, 0.3, 0.95],
            "Playground": [0.3, 0.8, 0.9],
            "Train": [0.3, 0.6, 0.95],
            "Auditorium": [0., 0., 0.],
            "Ballroom": [0.3, 0.3, 0.5],
            "Courtroom": [0., 0.2, 0.2],
            "Museum": [0.3, 0.3, 0.7],
            "Palace": [0.3, 0.3, 0.4],
            "Temple": [0.3, 0.5, 0.5],
        }
        
    elif args.dataset=='eth3d':
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]

            geo_mask_thres_all = {
                'courtyard': 1,
                'delivery_area':1,
                'electro':1,
                'facade':1,
                'kicker':1,
                'meadow':1,
                'office':1,
                'pipes':1,
                'playground':1,
                'relief':1,
                'relief_2':1,
                'terrace':1,
                'terrains':1,
                'botanical_garden':1,
                'boulders':1,
                'bridge':2,
                'door':1,
                'exhibition_hall':1,
                'lecture_room':1,
                'living_room':1,
                'lounge':1,
                'observatory':1,
                'old_computer':1,
                'statue':1,
                'terrace_2':1,
                }
            geo_pixel_thres_all = {
                'courtyard':0.5,
                'delivery_area':0.5,
                'electro':1,
                'facade':1,
                'kicker':1,
                'meadow':2,
                'office':2,
                'pipes':2,
                'playground':1,
                'relief':1,
                'relief_2':1,
                'terrace':0.5,
                'terrains':1,
                'botanical_garden':1,
                'boulders':0.5,
                'bridge':0.5,
                'door':0.5,
                'exhibition_hall':0.5,
                'lecture_room':0.5,
                'living_room':0.5,
                'lounge':2,
                'observatory':1,
                'old_computer':2,
                'statue':1,
                'terrace_2':0.5,
            }
    else:
        # general scene. we only have a single scene.
        testlist = [""]

    if args.save_depth:
        save_depth(testlist)

    # filtering
    if args.dataset != 'general':
        # benchmark datasets
        for scan in testlist:
            if args.dataset=='dtu':
                path = args.testpath
                pair_folder = os.path.join(args.testpath, scan)
                scan_id = int(scan[4:])
            else:
                pair_folder = os.path.join(args.testpath, scan)

            out_folder = os.path.join(args.outdir, scan)
            plypath = args.outdir + '/pc'
            if not os.path.exists(plypath):
                os.makedirs(plypath)

            if args.dataset == 'dtu':
                plyfilename = os.path.join(args.outdir, 'pc/mvs{:0>3}_l3.ply'.format(scan_id))
                filter_depth(
                    pair_folder,
                    out_folder,
                    plyfilename,
                    args.geo_mask_thres,
                    args.geo_pixel_thres,
                    args.geo_depth_thres,
                    args.photo_thres,
                    args.method,
                    args.dataset,
                )
            elif args.dataset == 'tank':
                scan = scan.split('/')[1]
                plyfilename = os.path.join(args.outdir, 'pc/{}.ply'.format(scan))
                filter_depth_dynamic(
                    scan,
                    pair_folder,
                    out_folder,
                    plyfilename,
                    photo_thres_all[scan],
                    args.method,
                )
            elif args.dataset == 'eth3d':
                plyfilename = os.path.join(args.outdir, 'pc/{}.ply'.format(scan))
                filter_depth(
                    pair_folder,
                    out_folder,
                    plyfilename,
                    geo_mask_thres_all[scan],
                    geo_pixel_thres_all[scan],
                    args.geo_depth_thres,
                    args.photo_thres,
                    args.method,
                    args.dataset,
                    )
    else:
        # demo
        pair_folder = args.testpath
        out_folder = args.outdir
        plyfilename = os.path.join(args.outdir, 'pc.ply')
        filter_depth(
            pair_folder,
            out_folder,
            plyfilename,
            args.geo_mask_thres,
            args.geo_pixel_thres,
            args.geo_depth_thres,
            args.photo_thres,
            args.method,
            args.dataset,
        )