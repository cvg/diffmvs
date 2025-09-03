import torch
from torch.utils.data import Dataset
from datasets.data_io import *
import os
import numpy as np
import cv2
from PIL import Image

class MVSDataset(Dataset):
    def __init__(
        self,
        datapath,
        n_views = 3,
        numdepth = 384,
        dataset = 'dtu',
        scan = ['scan1'],
        max_h = 4800,
        max_w = 6400,
    ):
        self.datapath = datapath
        self.dataset = dataset
        
        self.n_views = n_views
        self.input_scans = scan
        self.build_metas()
        self.numdepth = numdepth
        self.max_h, self.max_w = max_h, max_w

        if self.dataset == "dtu":
            self.img_wh = (1600,1152)
        elif self.dataset == "tank":
            self.img_wh = (1920,1056)
        elif self.dataset == "eth3d":
            self.img_wh = (1920,1280)
        
        if self.dataset == "general":
            self.cam_folder = "cams"
        else:
            self.cam_folder = "cams_1"

    def build_metas(self):
        self.metas = []
        self.scans = self.input_scans
        print("dataset", self.dataset, "scans", self.scans)

        if self.dataset == "general":
            with open(os.path.join(self.datapath, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = []
                    src_views_score = [float(x) for x in f.readline().rstrip().split()]
                    src_views_1 = src_views_score[1::2]
                    src_views_1 = [int(x) for x in src_views_1]
                    score = src_views_score[2::2]
                    for i in range(len(src_views_1)):
                        if score[i]>0.01 and src_views_1[i]!=ref_view:
                            src_views.append(src_views_1[i])
                    if len(src_views) != 0:
                        self.metas += [("", ref_view, src_views)]
        else:
            for scan in self.scans:
                print(os.path.join(self.datapath, scan, 'pair.txt'))
                with open(os.path.join(self.datapath, scan, 'pair.txt')) as f:
                    num_viewpoint = int(f.readline())
                    for _ in range(num_viewpoint):
                        ref_view = int(f.readline().rstrip())
                        src_views = []
                        src_views_score = [float(x) for x in f.readline().rstrip().split()]
                        src_views_1 = src_views_score[1::2]
                        src_views_1 = [int(x) for x in src_views_1]
                        score = src_views_score[2::2]
                        for i in range(len(src_views_1)):
                            if score[i]>0.1 and src_views_1[i]!=ref_view:
                                src_views.append(src_views_1[i])
                        if len(src_views) != 0:
                            self.metas += [(scan, ref_view, src_views)]

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))

        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[-1])
        if depth_min < 0:
            depth_min = 1.0
        return intrinsics, extrinsics, depth_min, depth_max

    def read_img(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32) / 255.
        original_h, original_w, _ = np_img.shape        
        return np_img, original_h, original_w

    def scale_img_fixed_size(self, img):
        """scale images to the given size"""
        img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_LINEAR)
        return img

    def scale_img_adaptive(self, img, intrinsics, max_w=6400, max_h=4800, base=32):
        """
        scale images based on maximum size and
        ensure that the images can be processed (divided by base) by the model
        """
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale_h = 1.0 * max_h / h
            scale_w = 1.0 * max_w / w
            new_w, new_h = scale_w * w // base * base, scale_h * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        scan, ref_view, src_views = self.metas[idx]
        view_ids = [ref_view] + src_views[:self.n_views-1]
        imgs = []
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            if self.dataset != "general":
                img_filename = os.path.join(self.datapath, scan, f'images/{vid:08d}.jpg')
                proj_filename = os.path.join(self.datapath, scan, self.cam_folder, f'{vid:08d}_cam.txt')
            else:
                img_filename = os.path.join(self.datapath, f'images/{vid:08d}.jpg')
                proj_filename = os.path.join(self.datapath, self.cam_folder, f'{vid:08d}_cam.txt')

            img, original_h, original_w = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_max = self.read_cam_file(proj_filename)

            if self.dataset != "general":
                img = self.scale_img_fixed_size(img)
                intrinsics[0] *= self.img_wh[0]/original_w
                intrinsics[1] *= self.img_wh[1]/original_h
            else:
                img, intrinsics = self.scale_img_adaptive(img, intrinsics,
                                                          self.max_w, self.max_h)
            imgs.append(img)
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                disp_min = 1. / depth_max
                disp_max = 1. / depth_min
                depth_values = np.linspace(disp_min, disp_max,
                                           self.numdepth, dtype=np.float32)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])

        proj_matrices = np.stack(proj_matrices)
        # 1/8 resolution
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.125
        # 1/4 resolution
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.25
        # 1/2 resolution
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5
        proj_matrices_ms = {
            "stage1": torch.from_numpy(stage1_pjmats.copy()).contiguous().float(),
            "stage2": torch.from_numpy(stage2_pjmats.copy()).contiguous().float(),
            "stage3": torch.from_numpy(stage3_pjmats.copy()).contiguous().float(),
            "stage4": torch.from_numpy(proj_matrices.copy()).contiguous().float(),
        }

        imgs = torch.from_numpy(imgs.copy()).contiguous().float()
        depth_values = torch.from_numpy(depth_values.copy()).contiguous().float()

        if self.dataset != "general":
            return {
                "imgs": imgs,                   
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"
            }
        else:
            return {
                "imgs": imgs,                   
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "filename": '{}/' + '{:0>8}'.format(view_ids[0]) + "{}"
            }
