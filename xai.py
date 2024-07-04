# Ensure Python 2 & 3 compatibility
from __future__ import print_function
# import the necessary packages
import time
start = time.time()

import pandas as pd
from datetime import timedelta
import argparse
from tqdm import tqdm
import matplotlib
import numpy as np
import os.path as path
import doe_tiff as dt
from osgeo import gdal
from osgeo.gdalconst import *

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import innvestigate

# Define defaults for input file parameters
CHANNELS = 5  # This will be redefined based on parameters
KERNEL_PIXELS = 17  # Default pixels by side on each tile
NUM_CLASSES = 2  # Default number of classes ("Geothermal", "Non-Geothermal")


def saveraster(inimage, outimage, data, datatype):
    inDs = gdal.Open(inimage)
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize
    driver = inDs.GetDriver()
    outDs = driver.Create(outimage, cols, rows, 1, datatype)
    outBand = outDs.GetRasterBand(1)
    outData = data.T
    outBand.WriteArray(outData, 0, 0)
    outBand.FlushCache()
    outBand.SetNoDataValue(-99)
    outDs.SetGeoTransform(inDs.GetGeoTransform())
    outDs.SetProjection(inDs.GetProjection())
    outDs.FlushCache()
    del outData
    outDs = None
    print("saving file: ", outimage)


print('Set-up complete.')

''' Main program '''
if __name__ == '__main__':
    ''' Main instructions '''
    print('Parsing input...')
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--channels", required=False, help='Number of channels in each image',
                    default=CHANNELS, type=int)
    ap.add_argument("-i", "--image", required=True,
                    help="path to input multi-band image (i.e., image file name)")
    ap.add_argument("-k", "--kernel_size", required=False,
                    help='Number of pixels by side in each image',
                    default=KERNEL_PIXELS, type=int)
    ap.add_argument("-l", "--layer", required=True, help="path to output raster layers without file extension")
    ap.add_argument("-m", "--model", required=True, help="path to input model")
    ap.add_argument("-n", "--num_classes", required=False, help='Number of classes',
                    default=NUM_CLASSES, type=int)
    ap.add_argument("-p", "--datacube", required=False, type=str, default="prediction_map.npy",
                    help="path to output data cube (ending in .npy)")
    ap.add_argument("-o", "--outputcsv", required=False, type=str, help="Raster file ending in .gri",
                    default="prediction_raster.gri")
    ap.add_argument("-x", "--explainable", required=True, type=str, help="Explainable AI method")

    args = vars(ap.parse_args())
    num_channels = args["channels"]
    image_name = args["image"]
    kernel_size = args["kernel_size"]
    layerfile = args["layer"]
    model_file = args["model"]
    num_classes = args["num_classes"]
    datacube = args["datacube"]
    outputcsv = args["outputcsv"]
    xaimethod = args["explainable"]
    limit = 1000
    # Ensures model file exists and is really a file
    PADDING = int(kernel_size / 2)
    try:
        assert path.exists(model_file), 'Model path {} does not exist'.format(model_file)
        assert path.isfile(model_file), 'Model file {} is not a file'.format(model_file)
        model_exist = True
    except:
        model_exist = False
        raise FileNotFoundError
    try:
        assert path.isfile(image_name), 'Image file {}: is not a file'.format(model_file)
        img_b = dt.io.read_gdal_file(image_name)
        max_channels = img_b.shape[2]
    except:
        print("Image file not found or erroneous", image_name)
        raise FileNotFoundError
    weights_exist = False
    # Read model file
    print('[INFO] Loading model from file...')

    # Define distribution strategy
    # strategy = tf.distribute.MirroredStrategy()

    # construct model under distribution strategy scope
    # with strategy.scope():

    model3 = tf.keras.models.load_model(model_file)
    model3.summary()
    model_wo_softmax = innvestigate.model_wo_softmax(model3)
    try:
        analyzer = innvestigate.create_analyzer(xaimethod, model_wo_softmax)
    except:
        print("ex")
    # Get rid of NaN's
    img_b = np.array(img_b, dtype=np.float64)
    img_b = np.nan_to_num(img_b)
    print("Image shape:", img_b.shape)
    assert num_channels > 0, 'Channels has to be a positive integer'
    assert num_channels <= max_channels, 'Channels has to be equal or lower than {}'.format(max_channels)
    img_b_scaled = img_b[:, :, 1:num_channels]
    print("DLM input image shape:", img_b_scaled.shape)
    mask_b = img_b[:, :, 0]  # first band is Ground Truth
    (img_x, img_y) = mask_b.shape
    print("Mask shape:", mask_b.shape)
    # new_map = np.zeros_like(mask_b)  # creates empty map
    new_map = np.empty((mask_b.shape[0], mask_b.shape[1], num_channels - 1))
    new_map[:] = np.nan
    print("Map shape:", new_map.shape)
    for i in range(0, img_b_scaled.shape[2]):
        print("band (", i, ") min:", img_b_scaled[:, :, i].min())
        print("band (", i, ") max:", img_b_scaled[:, :, i].max())
    img_b_scaled = dt.frame_image(img_b_scaled, PADDING)
    sc = dt.GeoTiffConvolution(img_b_scaled, kernel_size, kernel_size)

    IMAGE_DIMS = (kernel_size, kernel_size, num_channels)
    BATCH_DIMS = (None, kernel_size, kernel_size, num_channels)
    ### Check whether multi-gpu option was enabled
    print('[INFO] Creating prediction map...')
    count = 0
    data = []
    xx = []
    yy = []
    print("imgx", range(img_x))
    print("imgy", range(img_y))
    strategy = tf.distribute.MirroredStrategy()
    # construct model under distribution strategy scope
    with strategy.scope():
        for i in tqdm(range(img_x), desc="Predicting...", ascii=False, ncols=75):
            # for i in tqdm(range(1), desc="Predicting...", ascii=False, ncols=75):
            for j in range(img_y):
                image = sc.apply_mask(i + PADDING, j + PADDING)
                data.append(image)
                xx.append(i)
                yy.append(j)
                count = count + 1
                if count == limit:
                    data = np.array(data, dtype=np.float64)
                    data = np.nan_to_num(data)
                    a = analyzer.analyze(data)
                    weights = np.sum(a, axis=1)
                    weights = np.sum(weights, axis=1)
                    for k in range(len(weights)):
                        new_map[xx[k], yy[k], :] = weights[k]
                    data = []
                    xx = []
                    yy = []
                    count = 0
        if count != 0:
            data = np.array(data, dtype=np.float64)
            data = np.nan_to_num(data)
            a = analyzer.analyze(data)
            weights = np.sum(a, axis=1)
            weights = np.sum(weights, axis=1)
            for k in range(len(weights)):
                new_map[xx[k], yy[k], :] = weights[k]
        new_map = np.asarray(new_map)
        print('saving file:', datacube)
        f = open(datacube, 'wb')
        np.save(f, new_map)
        for k in range(num_channels - 1):
            y = k + 1
            saveraster(image_name, layerfile + str(k) + ".gri", new_map[:, :, k], GDT_Float32)
        w = np.nansum(new_map, axis=0)
        w = np.nansum(w, axis=0)
        print(w)
        dict_val = {"Weights": w}
        df_val = pd.DataFrame(dict_val)
        df_val.to_csv(outputcsv, index=True, index_label="Id")

end = time.time()
td = timedelta(seconds=int(end - start))
print("Total processing time", td)
