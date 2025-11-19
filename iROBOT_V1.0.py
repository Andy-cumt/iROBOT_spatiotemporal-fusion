#An improved ROBust OpTimization-based (iROBOT) fusion model for reliable spatiotemporal seamless remote sensing data reconstruction
#test Images: Landsat + MCD43A4

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
        dst = driver.Create(path, array.shape[2], array.shape[1], n_band, 6)
        for b in range(n_band):
            dst.GetRasterBand(b + 1).WriteArray(array[b, :, :])
    del dst


def save_imgf(array, path):
    """
    For Float
    """
    driver = gdal.GetDriverByName("GTiff")

    if array.dtype != np.float32:
        array = array.astype(np.float32)

    if len(array.shape) == 2:

        dst = driver.Create(path, array.shape[1], array.shape[0], 1, 6)  # 6 = GDT_Float32
        dst.GetRasterBand(1).WriteArray(array)
    else:

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
            # resized_x = F.interpolate(x, scale_factor=SCALE_FACTOR, mode='bilinear', align_corners=True)
            resized_x = F.interpolate(x, scale_factor=SCALE_FACTOR, mode='nearest')
            resized_x = torch.squeeze(resized_x)
            resized_arr = resized_x.numpy()
            resized_images.append(resized_arr)
        coarseimages = resized_images
    return fineimages, coarseimages, segimages, paths2


t_start = timer()

################ Setting ################

im_dir1 = 'E:/Testdata/L' # Path to your Landsat image folder
im_dir2 = 'E:/Testdata/M' # Path to your MCD43A4 image folder
im_dir3 = 'E:/Testdata/S' # Path to your segmented image folder
out_folder = 'E:/Testdata/P' # Path to the folder where you plan to store the generated images
SCALE_FACTOR = 8
target_index = 1
Band = 6
ROBOT_beta = 1
ROBOT_lambda1 = 2
ROBOT_thresh = [0.98, 1.02]
cloud_threshold = 0.5  # 50% cloud coverage threshold
avg_band_diff_th = 0.3 # Relative difference threshold


data_landsat0, data_modis0, data_seg0, paths2 = loaddata(im_dir1, im_dir2, im_dir3)

outname = 'iROBOT_' + os.path.basename(paths2[target_index])

data_landsat = np.array(data_landsat0)[:, 0:Band, :, :]
data_landsat_mask = np.array(data_landsat0)[:, -1, :, :]
data_modis = np.array(data_modis0)[:, 0:Band, :, :]
data_seg = np.squeeze(np.array(data_seg0)[:, :, :])

data_landsat.shape

# input_data_seg = data_seg[target_index - 1].squeeze() #For temporal segmented imagery
input_data_seg = data_seg.squeeze()  #For composite segmented imagery
max_nums = np.max(input_data_seg) + 1

input_landsat_series = np.delete(data_landsat, target_index, axis=0)
input_landsatmask_series = np.delete(data_landsat_mask, target_index, axis=0)


input_modis_series = np.delete(data_modis, target_index, axis=0)
input_modis_predict = data_modis[target_index]

################ Adaptive Auxiliary Information Gap-Filling ################

input_landsat_nearest_maskall = data_landsat_mask[target_index - 1] * data_landsat_mask[target_index + 1]  # 都无云
input_landsat_nearest_maskpart = (
            (data_landsat_mask[target_index - 1] + data_landsat_mask[target_index + 1]) > 0).astype(int)  # 一方有云
input_landsat_nearest_mask1 = (data_landsat_mask[target_index - 1] == 1) & (data_landsat_mask[target_index + 1] == 0)
input_landsat_nearest_mask2 = (data_landsat_mask[target_index - 1] == 0) & (data_landsat_mask[target_index + 1] == 1)


input_landsat_nearest0 = (data_landsat[target_index - 1] + data_landsat[target_index + 1]) / 2
input_landsat_nearest1 = input_landsat_nearest0 * input_landsat_nearest_maskall \
                         + data_landsat[target_index - 1] * input_landsat_nearest_mask1 \
                         + data_landsat[target_index + 1] * input_landsat_nearest_mask2
input_landsat_nearest = input_landsat_nearest1 * input_landsat_nearest_maskpart + abs(
    1 - input_landsat_nearest_maskpart) * input_modis_predict

x = torch.from_numpy(input_landsat_nearest)
x = x.unsqueeze(0).to(torch.float)

resized_x = F.interpolate(x, scale_factor=1 / SCALE_FACTOR, mode='nearest')
resized_x = F.interpolate(resized_x, scale_factor=SCALE_FACTOR, mode='nearest')
resized_x = torch.squeeze(resized_x)

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


input_modis_predict[mask] = input_landsat_nearest_M[mask]


################ Potentially low-quality information detection ################
################ Object-level matching and approximating via Lasso ################

num_series, num_band, H, W = input_landsat_series.shape

output_predict = np.zeros_like(input_landsat_nearest)
output_tmpCp = np.zeros_like(input_landsat_nearest)
output_coefs = np.zeros_like(input_landsat_nearest)
cnt = np.zeros([H, W], dtype=np.float32)  ## count overlap


for s in range(max_nums):
    seg_pos = np.where(input_data_seg == s)

    # === Extract cloud mask data for current segmented region ===
    pat_dataLD = input_landsat_series[:, :, seg_pos[0], seg_pos[1]].astype(np.float32)
    pat_dataMD = input_modis_series[:, :, seg_pos[0], seg_pos[1]].astype(np.float32)
    pat_nearL = input_landsat_nearest[:, seg_pos[0], seg_pos[1]].astype(np.float32)
    pat_predM = input_modis_predict[:, seg_pos[0], seg_pos[1]].astype(np.float32)

    # Get cloud mask data for current segmented region
    pat_landsat_mask = input_landsatmask_series[:, seg_pos[0], seg_pos[1]].astype(np.float32)

    # Initialize valid samples mask
    valid_samples_mask = np.ones(pat_dataLD.shape[0], dtype=bool)

    for i in range(pat_dataLD.shape[0]): # Iterate through each auxiliary image
        # 1. Check Landsat cloud coverage - exclude if cloud coverage is too high
        landsat_cloud_ratio = 1.0 - np.mean(pat_landsat_mask[i])  # Landsat cloud coverage ratio

        # Exclude sample if Landsat cloud coverage exceeds threshold
        if landsat_cloud_ratio > cloud_threshold:
            valid_samples_mask[i] = False
            print(f"Excluded auxiliary image {i} in segmented region {s}, Landsat cloud coverage too high: {landsat_cloud_ratio:.2f}")
            continue

        # 2. Check MODIS cloud coverage - MODIS value of 0 indicates cloud
        # Calculate cloud coverage for each MODIS band, take the maximum
        modis_cloud_ratios = []
        for b in range(pat_dataMD.shape[1]):
            modis_band = pat_dataMD[i, b]
            modis_cloud_ratio = np.mean(modis_band == 0)  # Ratio of pixels with value 0
            modis_cloud_ratios.append(modis_cloud_ratio)

        max_modis_cloud_ratio = max(modis_cloud_ratios) if modis_cloud_ratios else 0

        # Exclude sample if MODIS cloud coverage exceeds threshold
        if max_modis_cloud_ratio > cloud_threshold:
            valid_samples_mask[i] = False
            print(f"Excluded auxiliary image {i} in segmented region {s}, MODIS cloud coverage too high: {max_modis_cloud_ratio:.2f}")
            continue

        # 3. Create cloud-free mask - pixels where both images are cloud-free
        # Landsat mask: pat_landsat_mask[i] equals 1 indicates cloud-free
        # MODIS mask: all bands not equal to 0 indicates cloud-free
        modis_cloud_free = np.ones_like(pat_landsat_mask[i], dtype=bool)
        for b in range(pat_dataMD.shape[1]):
            modis_cloud_free = modis_cloud_free & (pat_dataMD[i, b] != 0)

        cloud_free_mask = (pat_landsat_mask[i] > 0) & modis_cloud_free

        # 4. Calculate Landsat-MODIS differences in cloud-free areas
        # Calculate relative differences for each band
        band_diffs = []
        for b in range(pat_dataLD.shape[1]):  # Iterate through each band
            # Use only cloud-free pixels
            landsat_band = pat_dataLD[i, b][cloud_free_mask]
            modis_band = pat_dataMD[i, b][cloud_free_mask]

            # Ensure sufficient data points for difference calculation
            if len(landsat_band) < 10:
                continue

            # Calculate relative difference
            diff = np.abs(landsat_band - modis_band) / (np.abs(modis_band) + np.abs(landsat_band) + 1e-6)
            avg_diff = np.mean(diff)
            band_diffs.append(avg_diff)

        # 5. Decide whether to exclude sample based on relative differences
        if len(band_diffs) > 0:
            avg_band_diff = np.mean(band_diffs)
            if avg_band_diff > avg_band_diff_th:  # Relative difference threshold
                valid_samples_mask[i] = False
                print(f"Excluded auxiliary image {i} in segmented region {s}, average relative difference: {avg_band_diff:.4f}")
        else:
            # Exclude sample if differences cannot be calculated
            valid_samples_mask[i] = False
            print(f"Excluded auxiliary image {i} in segmented region {s}, cannot calculate differences")

    # Use valid samples for subsequent processing
    if np.sum(valid_samples_mask) > 0:
        # Use only valid samples
        valid_pat_dataLD = pat_dataLD[valid_samples_mask]
        valid_pat_dataMD = pat_dataMD[valid_samples_mask]

        # Update num_series to valid sample count
        valid_num_series = valid_pat_dataLD.shape[0]

        # Use valid samples for subsequent LASSO regression
        colFDo = valid_pat_dataLD.reshape([valid_num_series, -1]).T
        colCDo = valid_pat_dataMD.reshape([valid_num_series, -1]).T
    else:

        colFDo = pat_dataLD.reshape([num_series, -1]).T
        colCDo = pat_dataMD.reshape([num_series, -1]).T
        valid_num_series = num_series



    blockH = pat_dataLD.shape[2]

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

    param = {
        "lambda1": ROBOT_lambda1,
        "numThreads": -1,
        "mode": 0,
        "pos": True,
    }
    alpha = spams.lasso(X, D, **param)
    coefRes = np.array(alpha.todense())


    tmpFp = (colFDo @ coefRes).T.reshape([-1, num_band, blockH])
    resMask = ~((ROBOT_thresh[0] < np.sum(coefRes, axis=0)) & (np.sum(coefRes, axis=0) < ROBOT_thresh[1]))

    tmpCp = (colCDo @ coefRes).T.reshape([-1, num_band, blockH])
    a = np.mean((pat_predM - tmpCp), axis=2)
    a_expanded = a[:, :, np.newaxis]
    tmpFp = tmpFp + a_expanded


    ## save the results
    output_predict[:, seg_pos[0], seg_pos[1]] = output_predict[:, seg_pos[0], seg_pos[1]] + np.clip(tmpFp, 0, 10000)

    cnt[seg_pos[0], seg_pos[1]] += 1
output_predict /= cnt

save_img(output_predict, os.path.join(out_folder, outname))

t_end = timer()
print(f'Time: {t_end - t_start}s')
