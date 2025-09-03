import os, cv2
import numpy as np
from datasets.data_io import read_pfm, read_camera_parameters, save_mask, read_pair_file, read_img
from plyfile import PlyData, PlyElement


# project the reference point cloud into the source view, then project back
def reproject_with_depth(
    depth_ref,
    intrinsics_ref,
    extrinsics_ref,
    depth_src,
    intrinsics_src,
    extrinsics_src
):
    """project the reference point cloud into the source view, then project back"""
    width, height = depth_ref.shape[1], depth_ref.shape[0]

    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    xyz_ref = np.matmul(
        np.linalg.inv(intrinsics_ref),
        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1])
    )

    # coordniate transformation
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src,
                                  interpolation=cv2.INTER_LINEAR)

    xyz_src = np.matmul(
        np.linalg.inv(intrinsics_src),
        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1])
    )

    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    depth_reproj = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    K_xyz_reprojected = np.where(K_xyz_reprojected == 0, 1e-5, K_xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    xy_reprojected = np.clip(xy_reprojected, -1e8, 1e8)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reproj, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(
    depth_ref,
    intrinsics_ref,
    extrinsics_ref,
    depth_src,
    intrinsics_src,
    extrinsics_src,
    ref_depth_max,
    ref_depth_min,
    geo_pixel_thres=1.0,
    geo_depth_thres=0.01,
    ):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reproj, x2d_reproj, y2d_reproj, x2d_src, y2d_src = reproject_with_depth(
        depth_ref,
        intrinsics_ref,
        extrinsics_ref,
        depth_src,
        intrinsics_src,
        extrinsics_src
    )

    dist = np.sqrt((x2d_reproj - x_ref) ** 2 + (y2d_reproj - y_ref) ** 2)

    depth_diff = np.abs(depth_reproj - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < geo_pixel_thres, 
                            relative_depth_diff < geo_depth_thres)
    mask2 = np.logical_and(depth_ref > ref_depth_min, depth_ref < ref_depth_max)
    mask = np.logical_and(mask, mask2)
    depth_reproj[~mask] = 0
    return mask, depth_reproj, x2d_src, y2d_src


def filter_depth(
    pair_folder,
    out_folder,
    plyfilename,
    geo_mask_thres=3,
    geo_pixel_thres=1.0,
    geo_depth_thres=0.01,
    photo_thres=[0.3, 0.5, 0.5],
    method='casdiffmvs',
    dataset='dtu',
):
    # the pair file
    pair_file = os.path.join(pair_folder, "pair.txt")
    pair_data = read_pair_file(pair_file, dataset)

    vertexs = []
    vertex_colors = []

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        ref_intrinsics, ref_extrinsics, depth_max, depth_min = read_camera_parameters(
            os.path.join(out_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the reference image
        ref_img = read_img(os.path.join(out_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        ref_depth_est = read_pfm(
            os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        
        """photometric filtering"""
        if method == 'casdiffmvs':
            confidence0 = read_pfm(
                os.path.join(out_folder, 'conf0/{:0>8}.pfm'.format(ref_view)))[0]
            confidence1 = read_pfm(
                os.path.join(out_folder, 'conf1/{:0>8}.pfm'.format(ref_view)))[0]
            confidence2 = read_pfm(
                os.path.join(out_folder, 'conf2/{:0>8}.pfm'.format(ref_view)))[0]

            photo_mask0 = confidence0 > photo_thres[0]
            photo_mask1 = confidence1 > photo_thres[1]
            photo_mask2 = confidence2 > photo_thres[2]
            photo_mask = photo_mask0 & photo_mask1 & photo_mask2
        else:
            confidence0 = read_pfm(
                os.path.join(out_folder, 'conf0/{:0>8}.pfm'.format(ref_view)))[0]
            confidence1 = read_pfm(
                os.path.join(out_folder, 'conf1/{:0>8}.pfm'.format(ref_view)))[0]

            photo_mask0 = confidence0 > photo_thres[0]
            photo_mask1 = confidence1 > photo_thres[1]
            photo_mask = photo_mask0 & photo_mask1

        """geometric filtering"""
        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []
        geo_mask_sum = 0

        for i, src_view in enumerate(src_views):
            src_intrinsics, src_extrinsics, _, _ = read_camera_parameters(
                os.path.join(out_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))

            src_depth_est = read_pfm(
                os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            geo_mask, depth_reproj, x2d_src, y2d_src = check_geometric_consistency(
                ref_depth_est,
                ref_intrinsics,
                ref_extrinsics,
                src_depth_est,
                src_intrinsics, 
                src_extrinsics,
                depth_max,
                depth_min,
                geo_pixel_thres,
                geo_depth_thres
            )
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reproj)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        geo_mask = geo_mask_sum >= geo_mask_thres

        final_mask = np.logical_and(photo_mask, geo_mask)
        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)),
            photo_mask
        )
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)),
            geo_mask
        )
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)),
            final_mask
        )

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(
            out_folder,
            ref_view,
            photo_mask.mean(),
            geo_mask.mean(),
            final_mask.mean())
        )

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        
        color = ref_img[valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs],
                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors],
                             dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def check_geometric_consistency_dynamic(
    depth_ref,
    intrinsics_ref,
    extrinsics_ref,
    depth_src,
    intrinsics_src,
    extrinsics_src,
    dh_pixel_dist_num
):
    """dynamic filtering for tanks & temples"""
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reproj, x2d_reproj, y2d_reproj, x2d_src, y2d_src = reproject_with_depth(
        depth_ref,
        intrinsics_ref,
        extrinsics_ref,
        depth_src,
        intrinsics_src,
        extrinsics_src
    )
    dist = np.sqrt((x2d_reproj - x_ref) ** 2 + (y2d_reproj - y_ref) ** 2)
    depth_diff = np.abs(depth_reproj - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    masks = []
    for i in range(dh_pixel_dist_num[0], 11):
        mask = np.logical_and(dist < i / dh_pixel_dist_num[1],
                              relative_depth_diff < i / dh_pixel_dist_num[2])
        masks.append(mask)
    depth_reproj[~mask] = 0

    return masks, mask, depth_reproj, x2d_src, y2d_src

def filter_depth_dynamic(
    scan,
    pair_folder,
    out_folder,
    plyfilename,
    photo_thres=[0.3, 0.5, 0.5],
    method='casdiffmvs',
    dataset='tank'
):
    """dynamic filtering for tanks & temples"""

    dh_view_num_all = {
        'Family':2, 'Francis':9, 'Horse':2,
        'Lighthouse':6, 'M60':4, 'Panther':3,
        'Playground':6, 'Train':3,
        'Auditorium':2, 'Ballroom':2, 'Courtroom':2,
        'Museum':2, 'Palace':2, 'Temple':1
    }
    dist_all = {
        'Family':12, 'Francis':8, 'Horse':4,
        'Lighthouse':8, 'M60':8, 'Panther':4,
        'Playground':8, 'Train':4,
        'Auditorium':4, 'Ballroom':4, 'Courtroom':4,
        'Museum':4, 'Palace':4, 'Temple':4
    }
    rel_diff_all = {
        'Family':1600, 'Francis':1600, 'Horse':1300,
        'Lighthouse':1600, 'M60':1600, 'Panther':1300,
        'Playground':1600, 'Train':1600,
        'Auditorium':1300, 'Ballroom':1300, 'Courtroom':1300,
        'Museum':1300, 'Palace':1300, 'Temple':1500
    }

    dh_view_num = dh_view_num_all[scan] 
    dh_dist = dist_all[scan]
    dh_rel_diff = rel_diff_all[scan]
    dh_pixel_dist_num = [dh_view_num, dh_dist, dh_rel_diff] 

    pair_file = os.path.join(pair_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)

    # for each reference view and the corresponding source views
    ct2 = -1
    for ref_view, src_views in pair_data:
        ct2 += 1
        # load the camera parameters
        ref_intrinsics, ref_extrinsics, ref_depth_max, ref_depth_min = read_camera_parameters(
            os.path.join(out_folder, 'cams/{:0>8}_cam.txt'.format(ref_view))
        )
        # load the reference image
        ref_img = read_img(os.path.join(out_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder,
                                              'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        # load the photometric mask of the reference view

        """photometric filtering"""
        if method == 'casdiffmvs':
            confidence0 = read_pfm(
                os.path.join(out_folder, 'conf0/{:0>8}.pfm'.format(ref_view)))[0]
            confidence1 = read_pfm(
                os.path.join(out_folder, 'conf1/{:0>8}.pfm'.format(ref_view)))[0]
            confidence2 = read_pfm(
                os.path.join(out_folder, 'conf2/{:0>8}.pfm'.format(ref_view)))[0]

            photo_mask0 = confidence0 > photo_thres[0]
            photo_mask1 = confidence1 > photo_thres[1]
            photo_mask2 = confidence2 > photo_thres[2]
            photo_mask = photo_mask0 & photo_mask1 & photo_mask2
        else:
            confidence0 = read_pfm(
                os.path.join(out_folder, 'conf0/{:0>8}.pfm'.format(ref_view)))[0]
            confidence1 = read_pfm(
                os.path.join(out_folder, 'conf1/{:0>8}.pfm'.format(ref_view)))[0]

            photo_mask0 = confidence0 > photo_thres[0]
            photo_mask1 = confidence1 > photo_thres[2] # use the last threshold for the final refinement
            photo_mask = photo_mask0 & photo_mask1

        """geometric filtering following D2HC-RMVSNet"""
        all_srcview_depth_ests = []
        geo_mask_sum = 0
        geo_mask_sums = []
        ct = 0
        for src_view in src_views:
            ct = ct + 1
            # camera parameters of the source view
            src_intrinsics, src_extrinsics, _, _  = read_camera_parameters(
                os.path.join(out_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # the estimated depth of the source view
            src_depth_est = read_pfm(
                os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            masks, geo_mask, depth_reproj, _, _ = check_geometric_consistency_dynamic(
                ref_depth_est,
                ref_intrinsics,
                ref_extrinsics,
                src_depth_est,
                src_intrinsics,
                src_extrinsics,
                dh_pixel_dist_num
            )

            if (ct == 1):
                for i in range(dh_view_num, 11):
                    geo_mask_sums.append(masks[i-dh_view_num].astype(np.int32))
            else:
                for i in range(dh_view_num, 11):
                    geo_mask_sums[i - dh_view_num] += masks[i-dh_view_num].astype(np.int32)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reproj)

        geo_mask = geo_mask_sum >= 10
        for i in range(dh_view_num, 11):
            geo_mask = np.logical_or(geo_mask, geo_mask_sums[i - dh_view_num] >= i)

        depth_est_averaged = ((sum(all_srcview_depth_ests) + ref_depth_est) /
                              (geo_mask_sum + 1))
        maskdepth = np.logical_and(depth_est_averaged >= ref_depth_min,
                                   depth_est_averaged <= ref_depth_max)

        final_mask = np.logical_and(photo_mask, geo_mask)
        final_mask = np.logical_and(final_mask, maskdepth)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), 
            photo_mask
        )
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)),
            geo_mask
        )
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), 
            final_mask
        )

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(
            out_folder,
            ref_view,
            photo_mask.mean(),
            geo_mask.mean(),
            final_mask.mean()
        ))

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        color = ref_img[:, :, :][valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs],
                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors],
                             dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)

