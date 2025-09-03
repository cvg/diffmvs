from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import *
import random
# the DTU dataset preprocessed by Yao Yao (only for training)

class MVSDataset(Dataset):
    def __init__(
        self,
        datapath,
        listfile,
        mode = "train",
        nviews = 5,
        ndepths = 384,
    ):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile

        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    if self.mode == "train":
                        for light_idx in range(7):
                            metas.append((scan, light_idx, ref_view, src_views))
                    else:
                        metas.append((scan, 3, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))

        depth_min = float(lines[11].split()[0])
        # hardcoded here following mvsnet_pytorch, depth_max is about 935
        interval_scale = 1.06 / (float(self.ndepths)/192.0)
        depth_interval = float(lines[11].split()[1]) * interval_scale
        depth_max = depth_interval * self.ndepths + depth_min
        return intrinsics, extrinsics, depth_min, depth_max

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def prepare_img(self, hr_img):
        #w1600-h1200-> 800-600 ; crop -> 640, 512
        h, w = hr_img.shape
        hr_img_ds = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)

        h, w = hr_img_ds.shape
        target_h, target_w = 512, 640
        start_h, start_w = (h - target_h)//2, (w - target_w)//2
        hr_img_crop = hr_img_ds[start_h: start_h + target_h,
                                start_w: start_w + target_w]
        return hr_img_crop

    def read_depth_mask(self, filename, mask_filename, depth_min, depth_max):
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_lr = self.prepare_img(depth_hr)

        mask = Image.open(mask_filename)
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 10).astype(np.float32)
        mask = self.prepare_img(mask)
        mask = mask.astype(np.bool_)
        mask = mask & (depth_lr>=depth_min) & (depth_lr<=depth_max)
        mask = mask.astype(np.float32)

        h, w = depth_lr.shape

        depth_lr_ms = {
            "stage1": cv2.resize(depth_lr, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_lr, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(depth_lr, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage4": depth_lr,
        }

        mask_ms = {
            "stage1": cv2.resize(mask, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(mask, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage4": mask,
        }

        return depth_lr_ms, mask_ms

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        # view_ids = [ref_view] + src_views[:self.nviews - 1]
        if self.mode == "train":
            src_views_ids = random.sample(src_views, self.nviews - 1)
        else:
            src_views_ids = src_views[:self.nviews - 1]
        view_ids = [ref_view] + src_views_ids

        imgs = []
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(
                self.datapath,
                'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx)
            )

            mask_filename_hr = os.path.join(
                self.datapath,
                'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid)
            )
            depth_filename_hr = os.path.join(
                self.datapath,
                'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid)
            )

            proj_mat_filename = os.path.join(self.datapath,
                                             'Cameras/train/{:0>8}_cam.txt').format(vid)

            img = self.read_img(img_filename)

            intrinsics, extrinsics, depth_min, depth_max = self.read_cam_file(proj_mat_filename)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_ms, mask_ms = self.read_depth_mask(
                    depth_filename_hr,
                    mask_filename_hr,
                    depth_min,
                    depth_max
                )
                disp_min = 1 / depth_max
                disp_max = 1 / depth_min
                depth_values = np.linspace(disp_min, disp_max, self.ndepths, dtype=np.float32)

            imgs.append(img)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])

        proj_matrices = np.stack(proj_matrices)
        stage1_pjmats = proj_matrices.copy()
        # original intrinsic matrix is for 1/4 resolution
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :]
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage4_pjmats = proj_matrices.copy()
        stage4_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        proj_matrices_ms = {
            "stage1": stage1_pjmats,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats,
            "stage4": stage4_pjmats,
        }

        return {
            "imgs": imgs,
            "proj_matrices": proj_matrices_ms,
            "depth": depth_ms,
            "depth_values": depth_values,
            "mask": mask_ms,
        }