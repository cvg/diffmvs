from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import random
import  cv2

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
        self.mode = mode
        self.listfile = listfile
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
            pair_file = "{}/cams/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) < self.nviews - 1:
                        print('less ref_view small {}'.format(self.nviews - 1))
                        continue
                    metas.append((scan, ref_view, src_views))
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
        depth_max = float(lines[11].split()[-1])
        return intrinsics, extrinsics, depth_min, depth_max

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        depth_image = np.array(read_pfm(filename)[0], dtype=np.float32)
        return depth_image

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        if self.mode == "train":
            src_views_ids = random.sample(src_views, self.nviews - 1)
        else:
            src_views_ids = src_views[:self.nviews - 1]
        view_ids = [ref_view] + src_views_ids

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        depth_ms = {}
        mask_ms = {}
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(
                self.datapath,
                '{}/blended_images/{:0>8}.jpg'.format(scan, vid)
            )
            proj_mat_filename = os.path.join(
                self.datapath,
                '{}/cams/{:0>8}_cam.txt'.format(scan, vid)
            )
            depth_filename = os.path.join(
                self.datapath,
                '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid)
            )

            imgs.append(self.read_img(img_filename).transpose([2, 0, 1]))
            intrinsics, extrinsics, depth_min, depth_max = self.read_cam_file(
                proj_mat_filename
            )

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                disp_min = 1 / depth_max
                disp_max = 1 / depth_min
                depth_values = np.linspace(disp_min, disp_max, self.ndepths, endpoint=False)
                depth_values = depth_values.astype(np.float32)

                depth = self.read_depth(depth_filename)
                h, w = depth.shape
                depth_ms = {
                    "stage1": cv2.resize(depth, (w // 8, h // 8),
                                         interpolation=cv2.INTER_NEAREST),
                    "stage2": cv2.resize(depth, (w // 4, h // 4),
                                         interpolation=cv2.INTER_NEAREST),
                    "stage3": cv2.resize(depth, (w // 2, h // 2),
                                         interpolation=cv2.INTER_NEAREST),
                    "stage4": depth,
                }
                mask_ms = {
                    "stage1": np.array((depth_ms["stage1"] >= depth_min) & (depth_ms["stage1"] <= depth_max), dtype=np.float32),
                    "stage2": np.array((depth_ms["stage2"] >= depth_min) & (depth_ms["stage2"] <= depth_max), dtype=np.float32),
                    "stage3": np.array((depth_ms["stage3"] >= depth_min) & (depth_ms["stage3"] <= depth_max), dtype=np.float32),
                    "stage4": np.array((depth_ms["stage4"] >= depth_min) & (depth_ms["stage4"] <= depth_max), dtype=np.float32),
                }

        # imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        proj_matrices = np.stack(proj_matrices)
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 8.0
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4.0
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2.0
        stage4_pjmats = proj_matrices.copy()
        stage4_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :]

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
