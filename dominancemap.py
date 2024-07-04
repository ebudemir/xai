import argparse

import numpy as np
from osgeo import gdal
from tqdm import tqdm
import os


def createDominance(predictionmap, xairesult, dominancemap):
    ds = gdal.Open(predictionmap)
    band = ds.GetRasterBand(1)
    p_data = band.ReadAsArray()
    dt = np.load(xairesult)
    dt = np.array(dt, dtype=np.float32)
    data_m = dt[:, :, 0]
    data_m[p_data != 1] = np.nan
    data_t = dt[:, :, 1]
    data_t[p_data != 1] = np.nan
    data_f = dt[:, :, 2]
    data_f[p_data != 1] = np.nan
    data = np.empty(data_m.shape)
    data_mf = np.max(np.stack((data_m, data_f)), axis=0)
    data_ft = np.max(np.stack((data_t, data_f)), axis=0)
    data_mt = np.max(np.stack((data_m, data_t)), axis=0)
    data[data_m > data_ft] = 1
    data[data_t > data_mf] = 2
    data[data_f > data_mt] = 3
    data[(data_f == data_t) & (data_f > data_m)] = 4
    data[(data_f == data_m) & (data_f > data_t)] = 5
    data[(data_m == data_t) & (data_t > data_f)] = 6
    data[(data_f == data_t) & (data_f == data_m)] = 7
    data[p_data != 1] = np.nan

    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    gettr = ds.GetGeoTransform()
    crs = ds.GetProjection()

    target_ds = gdal.GetDriverByName('GTiff').Create(dominancemap, xsize, ysize, 1, gdal.GDT_Int32)
    target_ds.SetGeoTransform(gettr)
    target_ds.SetProjection(crs)
    target_band = target_ds.GetRasterBand(1)
    target_band.WriteArray(data, 0, 0)
    target_band.FlushCache()
    target_ds = None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prediction", required=True, help='Prediction map (.gri)')
    ap.add_argument("-x", "--xairesult", required=True, help="xai result file (.npy)")
    ap.add_argument("-d", "--dominancemap", required=False, help='dominance map (.gri)')
    args = vars(ap.parse_args())
    pred = args["prediction"]
    data = args["xairesult"]
    result = args["dominancemap"]
    createDominance(pred, data, result)
