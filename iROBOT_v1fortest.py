#An Improved ROBust OpTimization-based (iROBOT) Fusion Model for Reliable Spatiotemporal Seamless Remote Sensing Data Reconstruction
# DizhouGuo@163.com
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import spams
import rasterio
from pathlib import Path
import torch.nn.functional as F
import torch
from osgeo import gdal
from skimage.exposure import match_histograms
from timeit import default_timer as timer


def save_img(array, path):
    driver = gdal.GetDriverByName("GTiff")
    if len(array.shape) == 2:
        dst = driver.Create(path, array.shape[1], array.shape[0], 1, 2)
        dst.GetRasterBand(1).WriteArray(array)
    else:
        # save all bands
        n_band = array.shape[0]
        dst = driver.Create(path, array.shape[1], array.shape[2], n_band, 6)
        for b in range(n_band):
            dst.GetRasterBand(b + 1).WriteArray(array[b, :, :])
    del dst


def save_imgf(array, path):
    """
    改进的保存函数，确保所有数据都保存为浮点数格式
    """
    driver = gdal.GetDriverByName("GTiff")

    # 确保数据类型为float32
    if array.dtype != np.float32:
        array = array.astype(np.float32)

    if len(array.shape) == 2:
        # 单波段：使用Float32数据类型（代码6）
        dst = driver.Create(path, array.shape[1], array.shape[0], 1, 6)  # 6 = GDT_Float32
        dst.GetRasterBand(1).WriteArray(array)
    else:
        # 多波段：使用Float32数据类型
        n_band = array.shape[0]
        dst = driver.Create(path, array.shape[2], array.shape[1], n_band, 6)  # 6 = GDT_Float32
        for b in range(n_band):
            dst.GetRasterBand(b + 1).WriteArray(array[b, :, :])
    del dst


def normC2(X):  ##[feature, Num]
    N_feature, N_num = X.shape
    imMIN = np.mean(np.min(X, axis=0))
    imMAX = np.mean(np.max(X, axis=0))
    res = (X - imMIN) / (imMAX - imMIN + 1e-6)
    divd = N_feature * 100
    return np.asfortranarray(res / divd, dtype=np.float64), (imMIN, imMAX, divd)


#### 1. read data

def loaddata(im_dir1, im_dir2, im_dir3):
    paths1 = []
    paths2 = []
    paths3 = []

    for path in Path(im_dir1).glob('*.tif'):
        paths1.append(path.expanduser().resolve())  # 获得目录

    for path2 in Path(im_dir2).glob('*.tif'):
        paths2.append(path2.expanduser().resolve())  # 获得目录

    for path3 in Path(im_dir3).glob('*.tif'):
        paths3.append(path3.expanduser().resolve())  # 获得目录

    assert len(paths1) == len(paths2)

    fineimages = []
    coarseimages = []
    segimages = []

    for p in paths1:
        with rasterio.open(str(p)) as ds:
            im = ds.read()
            fineimages.append(im)
    for p in paths2:
        with rasterio.open(str(p)) as ds:
            im = ds.read()
            coarseimages.append(im)
    # segimages = im_dir3.read()
    for p in paths3:
        with rasterio.open(str(p)) as ds:
            im = ds.read()
            segimages.append(im)

    if coarseimages[0].shape[1] * SCALE_FACTOR == fineimages[0].shape[1]:
        resized_images = []
        for arr in coarseimages:
            x = torch.from_numpy(arr)
            x = x.unsqueeze(0).to(torch.float)
            # 对张量进行插值
            # resized_x = F.interpolate(x, scale_factor=SCALE_FACTOR, mode='bilinear', align_corners=True)
            resized_x = F.interpolate(x, scale_factor=SCALE_FACTOR, mode='nearest')
            resized_x = torch.squeeze(resized_x)
            # 将 PyTorch 张量转换回 NumPy 数组
            resized_arr = resized_x.numpy()

            # 将插值后的数组添加到列表中
            resized_images.append(resized_arr)
        coarseimages = resized_images
    return fineimages, coarseimages, segimages, paths2


t_start = timer()
# 这个就是原版本
im_dir1 = 'E:/idea_paper/OL-ROBOT/XJ/L2'
im_dir2 = 'E:/idea_paper/OL-ROBOT/XJ/M2'
im_dir3 = 'E:/idea_paper/OL-ROBOT/XJ/S2'
out_folder = 'E:/idea_paper/OL-ROBOT/XJ\\testtime'
SCALE_FACTOR = 16
#Time: 203.7721764s XJ 13 15.4s
#Time: 198.1175178s PY 15，13.2s
for target_index in range(1, 14):
    # target_index = 1

    # data_landsat,data_modis = loaddata(im_dir1,im_dir2)
    data_landsat0, data_modis0, data_seg0, paths2 = loaddata(im_dir1, im_dir2, im_dir3)

    outname = 'OLROBOT3' + os.path.basename(paths2[target_index])
    # outname = 'OLROBOT'+os.path.basename(paths2[target_index])
    # outname_MP = 'Mp_'+os.path.basename(paths2[target_index])
    # outname_FN = 'Fn_'+os.path.basename(paths2[target_index])

    # data_landsat = np.array(data_landsat0)[:,0:6,:,:]
    # data_landsat_mask = np.array(data_landsat0)[:,-1,:,:]
    # data_modis = np.array(data_modis0)[:,0:6,:,:]
    # data_seg = np.squeeze(np.array(data_seg0)[:,:,:])

    data_landsat = np.array(data_landsat0)[:, 0:6, :, :]
    data_landsat_mask = np.array(data_landsat0)[:, -1, :, :]
    data_modis = np.array(data_modis0)[:, 0:6, :, :]
    data_seg = np.squeeze(np.array(data_seg0)[:, :, :])

    data_landsat.shape

    #### 2. specify data settings

    ###注意这边设置分割目标

    input_data_seg = data_seg[target_index - 1].squeeze()
    # input_data_seg = data_seg.squeeze() #针对单个的！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    max_nums = np.max(input_data_seg) + 1

    input_landsat_series = np.delete(data_landsat, target_index, axis=0)
    input_landsatmask_series = np.delete(data_landsat_mask, target_index, axis=0)
    # for img_idx in range(input_landsat_series.shape[0]):  # 遍历每个影像
    #     for band_idx in range(input_landsat_series.shape[1]):  # 遍历每个波段
    #         input_landsat_series[img_idx, band_idx, :, :] = input_landsat_series[img_idx, band_idx, :, :] * input_landsatmask_series[img_idx, :, :]

    input_modis_series = np.delete(data_modis, target_index, axis=0)
    input_modis_predict = data_modis[target_index]

    input_landsat_nearest_maskall = data_landsat_mask[target_index - 1] * data_landsat_mask[target_index + 1]  # 都无云
    input_landsat_nearest_maskpart = (
                (data_landsat_mask[target_index - 1] + data_landsat_mask[target_index + 1]) > 0).astype(int)  # 一方有云
    input_landsat_nearest_mask1 = (data_landsat_mask[target_index - 1] == 1) & (data_landsat_mask[target_index + 1] == 0)
    input_landsat_nearest_mask2 = (data_landsat_mask[target_index - 1] == 0) & (data_landsat_mask[target_index + 1] == 1)

    # 辅助参考影像的合成，无云的部分为两时相间平均值，否则用MODIS替代
    input_landsat_nearest0 = (data_landsat[target_index - 1] + data_landsat[target_index + 1]) / 2
    input_landsat_nearest1 = input_landsat_nearest0 * input_landsat_nearest_maskall \
                             + data_landsat[target_index - 1] * input_landsat_nearest_mask1 \
                             + data_landsat[target_index + 1] * input_landsat_nearest_mask2
    input_landsat_nearest = input_landsat_nearest1 * input_landsat_nearest_maskpart + abs(
        1 - input_landsat_nearest_maskpart) * input_modis_predict

    x = torch.from_numpy(input_landsat_nearest)
    x = x.unsqueeze(0).to(torch.float)
    # 对张量进行插值
    # resized_x = F.interpolate(x, scale_factor=SCALE_FACTOR, mode='bilinear', align_corners=True)
    resized_x = F.interpolate(x, scale_factor=1 / 16, mode='nearest')
    resized_x = F.interpolate(resized_x, scale_factor=16, mode='nearest')
    resized_x = torch.squeeze(resized_x)
    # 将 PyTorch 张量转换回 NumPy 数组
    resized_input_landsat_nearest = resized_x.numpy()
    #
    input_landsat_nearest_M = np.zeros_like(resized_input_landsat_nearest)
    mask = input_modis_predict == 0
    for band in range(resized_input_landsat_nearest.shape[0]):
        mask_avg = np.mean(mask[band])
        if mask_avg >= 0.9:
            input_landsat_nearest_M[band] = match_histograms(resized_input_landsat_nearest[band],
                                                             input_modis_predict[band])
        else:
            input_landsat_nearest_M[band] = input_landsat_nearest[band]

        # 创建一个掩膜，标记 input_modis_predict 中值为0的位置
    # mask = input_modis_predict == 0
    # 使用掩膜来选择 input_landsat_nearest 中对应位置的值，并替换 input_modis_predict 中的0
    input_modis_predict[mask] = input_landsat_nearest_M[mask]


    #### 3. ROBOT
    ## 3.1 set parameters

    ROBOT_beta = 1
    ROBOT_lambda1 = 2
    # bSize = 30
    # step = 20
    ROBOT_thresh = [0.98, 1.02]

    ## 3.2 spatiotemporal fusion
    num_series, num_band, H, W = input_landsat_series.shape

    output_predict = np.zeros_like(input_landsat_nearest)
    output_tmpCp = np.zeros_like(input_landsat_nearest)
    output_coefs = np.zeros_like(input_landsat_nearest)
    cnt = np.zeros([H, W], dtype=np.float32)  ## count overlap

    # 新增：初始化alpha权值矩阵
    # 创建一个数组来存储每个辅助影像的回归权值
    # 形状为 (num_series, H, W)，每个像素位置存储对应辅助影像的权值
    # alpha_weights = np.zeros((num_series, H, W), dtype=np.float32)
    # alpha_weights_count = np.zeros((H, W), dtype=np.float32)  # 用于计数

    for s in range(max_nums):
        seg_pos = np.where(input_data_seg == s)

        # === 修改开始：获取当前分割区域的云掩膜数据 ===
        pat_dataLD = input_landsat_series[:, :, seg_pos[0], seg_pos[1]].astype(np.float32)
        pat_dataMD = input_modis_series[:, :, seg_pos[0], seg_pos[1]].astype(np.float32)
        pat_nearL = input_landsat_nearest[:, seg_pos[0], seg_pos[1]].astype(np.float32)
        pat_predM = input_modis_predict[:, seg_pos[0], seg_pos[1]].astype(np.float32)

        # 获取当前分割区域的云掩膜数据
        pat_landsat_mask = input_landsatmask_series[:, seg_pos[0], seg_pos[1]].astype(np.float32)

        # 初始化有效样本掩膜
        valid_samples_mask = np.ones(pat_dataLD.shape[0], dtype=bool)

        for i in range(pat_dataLD.shape[0]):  # 遍历每个辅助影像
            # 1. 检查Landsat云覆盖率 - 如果云覆盖率太高，直接排除
            landsat_cloud_ratio = 1.0 - np.mean(pat_landsat_mask[i])  # Landsat云覆盖率

            # 如果Landsat云覆盖率超过阈值，排除该样本
            cloud_threshold = 0.5  # 50%云覆盖率阈值
            if landsat_cloud_ratio > cloud_threshold:
                valid_samples_mask[i] = False
                print(f"分割区域 {s} 中排除了辅助影像 {i}，Landsat云覆盖率过高: {landsat_cloud_ratio:.2f}")
                continue

            # 2. 检查MODIS云覆盖率 - MODIS值为0表示有云
            # 计算MODIS每个波段的云覆盖率，取最大值
            modis_cloud_ratios = []
            for b in range(pat_dataMD.shape[1]):
                modis_band = pat_dataMD[i, b]
                modis_cloud_ratio = np.mean(modis_band == 0)  # 值为0的像素比例
                modis_cloud_ratios.append(modis_cloud_ratio)

            max_modis_cloud_ratio = max(modis_cloud_ratios) if modis_cloud_ratios else 0

            # 如果MODIS云覆盖率超过阈值，排除该样本
            if max_modis_cloud_ratio > cloud_threshold:
                valid_samples_mask[i] = False
                print(f"分割区域 {s} 中排除了辅助影像 {i}，MODIS云覆盖率过高: {max_modis_cloud_ratio:.2f}")
                continue

            # 3. 创建无云掩膜 - 两个影像都无云的像素
            # Landsat掩膜：pat_landsat_mask[i] 为1表示无云
            # MODIS掩膜：所有波段都不为0表示无云
            modis_cloud_free = np.ones_like(pat_landsat_mask[i], dtype=bool)
            for b in range(pat_dataMD.shape[1]):
                modis_cloud_free = modis_cloud_free & (pat_dataMD[i, b] != 0)

            cloud_free_mask = (pat_landsat_mask[i] > 0) & modis_cloud_free



            # 4. 在无云区域计算Landsat与MODIS的差异
            # 计算每个波段的相对差异
            band_diffs = []
            for b in range(pat_dataLD.shape[1]):  # 遍历每个波段
                # 只使用无云像素
                landsat_band = pat_dataLD[i, b][cloud_free_mask]
                modis_band = pat_dataMD[i, b][cloud_free_mask]

                # 确保有足够的数据点计算差异
                if len(landsat_band) < 10:
                    continue

                # 计算相对差异
                diff = np.abs(landsat_band - modis_band) / (np.abs(modis_band) + np.abs(landsat_band) + 1e-6)
                avg_diff = np.mean(diff)
                band_diffs.append(avg_diff)

            # 5. 基于相对差异决定是否排除样本
            if len(band_diffs) > 0:
                avg_band_diff = np.mean(band_diffs)
                if avg_band_diff > 0.3:  # 相对差异阈值
                    valid_samples_mask[i] = False
                    print(f"分割区域 {s} 中排除了辅助影像 {i}，平均相对差异: {avg_band_diff:.4f}")
            else:
                # 如果无法计算差异，排除该样本
                valid_samples_mask[i] = False
                print(f"分割区域 {s} 中排除了辅助影像 {i}，无法计算差异")

        # 使用有效样本进行后续处理
        if np.sum(valid_samples_mask) > 0:
            # 只使用有效样本
            valid_pat_dataLD = pat_dataLD[valid_samples_mask]
            valid_pat_dataMD = pat_dataMD[valid_samples_mask]

            # 更新num_series为有效样本数量
            valid_num_series = valid_pat_dataLD.shape[0]

            # 使用有效样本进行后续的LASSO回归
            colFDo = valid_pat_dataLD.reshape([valid_num_series, -1]).T
            colCDo = valid_pat_dataMD.reshape([valid_num_series, -1]).T
        else:
            # 如果没有有效样本，使用所有样本
            # print(f"分割区域 {s} 没有有效样本，使用所有样本")
            colFDo = pat_dataLD.reshape([num_series, -1]).T
            colCDo = pat_dataMD.reshape([num_series, -1]).T
            valid_num_series = num_series
        # === 修改结束 ===
        # colFDo = pat_dataLD.reshape([num_series, -1]).T
        # colCDo = pat_dataMD.reshape([num_series, -1]).T


        blockH = pat_dataLD.shape[2]
        # blockW = pat_dataLD.shape[3]

        colCp = pat_predM.reshape([1, -1]).T
        colFr = pat_nearL.reshape([1, -1]).T

        ## 2. Normalization
        colFD, statsF = normC2(colFDo)
        colCD, statsC = normC2(colCDo)

        ## 3. Fusion via Optimization
        imMIN, imMAX, divd = statsC
        colCp = np.asfortranarray((colCp - imMIN) / (imMAX - imMIN + 1e-6) / divd, dtype=np.float64)
        imMIN, imMAX, divd = statsF
        colFr = np.asfortranarray((colFr - imMIN) / (imMAX - imMIN + 1e-6) / divd, dtype=np.float64)

        X = np.vstack([colCp, colFr * ROBOT_beta])
        D = np.vstack([colCD, colFD * ROBOT_beta])
        # X = np.vstack([colFr * ROBOT_beta])
        # D = np.vstack([colFD*ROBOT_beta])
        param = {
            "lambda1": ROBOT_lambda1,
            "numThreads": -1,
            "mode": 0,
            "pos": True,
        }
        alpha = spams.lasso(X, D, **param)
        coefRes = np.array(alpha.todense())
        ###############################################################################
        # ## 新增：保存每个辅助影像的回归权值
        # # coefRes的形状是 (num_series, 当前分割区域的像素数)
        # # 我们需要将这些权值分配到对应的空间位置
        # # === 修改：需要处理有效样本的情况 ===
        # if np.sum(valid_samples_mask) > 0:
        #     # 有样本被排除的情况
        #     valid_coef_index = 0
        #     for i in range(num_series):
        #         if valid_samples_mask[i]:
        #             # 只对有效样本分配权值
        #             alpha_weights[i, seg_pos[0], seg_pos[1]] += coefRes[valid_coef_index, :]
        #             valid_coef_index += 1
        #         else:
        #             # 被排除的样本权值为0
        #             alpha_weights[i, seg_pos[0], seg_pos[1]] += 0
        # else:
        #     # 没有样本被排除的情况
        #     for i in range(num_series):
        #         alpha_weights[i, seg_pos[0], seg_pos[1]] += coefRes[i, :]
        #
        # # 更新计数
        # alpha_weights_count[seg_pos[0], seg_pos[1]] += 1
        ###############################################################################
        ## reconstruct images using the obtained coefficients
        # tmpFp = (colFDo @ coefRes).T.reshape([-1, num_band, blockH, blockW])
        tmpFp = (colFDo @ coefRes).T.reshape([-1, num_band, blockH])
        resMask = ~((ROBOT_thresh[0] < np.sum(coefRes, axis=0)) & (np.sum(coefRes, axis=0) < ROBOT_thresh[1]))
        # if np.sum(resMask) > 0: ## distributing residuals
        # tmpCp = (colCDo @ coefRes).T.reshape([-1, num_band, blockH, blockW])
        tmpCp = (colCDo @ coefRes).T.reshape([-1, num_band, blockH])
        a = np.mean((pat_predM - tmpCp), axis=2)
        a_expanded = a[:, :, np.newaxis]
        tmpFp = tmpFp + a_expanded
        # tmpFp = tmpFp + (pat_predM - tmpCp)

        ## save the results
        output_predict[:, seg_pos[0], seg_pos[1]] = output_predict[:, seg_pos[0], seg_pos[1]] + np.clip(tmpFp, 0, 10000)
        # output_tmpCp[:, seg_pos[0], seg_pos[1]] = np.clip(tmpCp, 0, 10000)
        # output_coefs[:, seg_pos[0], seg_pos[1]] = np.sum(coefRes, axis=0)
        cnt[seg_pos[0], seg_pos[1]] += 1
    output_predict /= cnt
    # output_tmpCp /= cnt
    # output_coefs /= cnt
    save_img(output_predict, os.path.join(out_folder, outname))
# save_img(input_modis_predict, os.path.join(out_folder, outname_MP))
# save_img(input_landsat_nearest, os.path.join(out_folder, outname_FN))
t_end = timer()
print(f'Time: {t_end - t_start}s')

# # 新增：对alpha权值进行平均处理
# # 避免除以0
# alpha_weights_count[alpha_weights_count == 0] = 1
# for i in range(num_series):
#     alpha_weights[i] /= alpha_weights_count
#
# # 新增：保存alpha权值
# # 保存每个辅助影像的权值图
# for i in range(num_series):
#     alpha_outname = f'alpha_weights_series_{i + 1}.tif'
#     save_imgf(alpha_weights[i], os.path.join(out_folder, alpha_outname))
#     print(f"Saved alpha weights for series {i + 1}")
# # 新增：保存所有辅助影像权值的总和（用于可视化）
# alpha_sum = np.sum(alpha_weights, axis=0)
# save_img(alpha_sum, os.path.join(out_folder, 'alpha_weights_sum.tif'))
#
# # 新增：保存每个辅助影像权值的统计信息
# print("\nAlpha weights statistics:")
# for i in range(num_series):
#     alpha_mean = np.mean(alpha_weights[i])
#     alpha_std = np.std(alpha_weights[i])
#     alpha_max = np.max(alpha_weights[i])
#     alpha_min = np.min(alpha_weights[i])
#     print(f"Series {i + 1}: mean={alpha_mean:.4f}, std={alpha_std:.4f}, min={alpha_min:.4f}, max={alpha_max:.4f}")

# outname3 = 'output_tmpCp2.tif'
# outname4 = 'output_coefs.tif'
# save_img(output_tmpCp, os.path.join(out_folder, outname3))
# save_img(output_coefs, os.path.join(out_folder, outname4))

#### 4. Accuracy assessment
# print("Accuracy assessment")
# true_img = data_landsat[target_index] / 10000
# pred_img = output_predict             / 10000
# print("RMSE", np.sqrt(np.mean( (true_img - pred_img)**2 , axis=(1,2))))
# print("MAE ", np.mean(np.abs(true_img - pred_img), axis=(1,2)))
# print("CC  ", np.array([np.corrcoef(true_img[b].ravel(), pred_img[b].ravel())[0,1] for b in range(num_band)]))