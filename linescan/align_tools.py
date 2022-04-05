from sklearn import preprocessing
from scipy import signal
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy import ndimage as ndi
import skimage
from skimage import io, filters, morphology
import pandas as pd
import glob
import random
import glob
import seaborn as sns
from scipy import stats
from scipy.signal import chirp, find_peaks, peak_widths
import pandas as pd
from pathlib import Path
import scipy.stats
from read_roi import read_roi_zip


def linescan(
    image_path,
    roi_path,
    channels,
    number_of_channels,
    align_channel,
    align_method,
    normalize=True,
    pixelsize=1,
    align=True,
):
    if (number_of_channels == 2) and (align_method == "half_max") :
        result_df = linescan_half_alingne_lagacy_2c_dnafirst(
            image_path,
            roi_path,
            channels,
            number_of_channels=2,
            #This is hacked so bad
            align_channel=align_channel,
            normalize=True,
            pixelsize=pixelsize,
            align=align,
        )
        #print(result_df)
        return result_df
    elif (number_of_channels == 3) and (align_method == "half_max"):
        result_df = linescan_half_alingne_lagacy_3c(
            image_path,
            roi_path,
            channels,
            number_of_channels=3,
            align_channel=align_channel,
            normalize=True,
            pixelsize=pixelsize,
            align=align,
        )
        #print(result_df)
        return result_df

def linescan_half_alingne_lagacy_3c(
    image_path,
    roi_path,
    channels,
    number_of_channels=3,
    normalize=True,
    pixelsize=1,
    align_channel=1,
    align=True,
):
    # get roi and image
    scaling = 0.03525845591290619

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    # create plot canvas
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    image_peaks = [[], [], []]
    for single_image, single_roi in zip(image_path, roi_path):
        roi = read_roi_zip(single_roi)
        image = io.imread(single_image)

        # generate a color map iterator
        cmap = plt.get_cmap("tab10")
        colors = iter(cmap.colors)

        for channel in range(number_of_channels):
            newcolor = next(colors)
            # print(newcolor)
            channel_max = []
            # scaling=1
            for key, item in roi.items():
                img_slice = item["position"]["slice"]
                src = (item["y1"], item["x1"])
                dst = (item["y2"], item["x2"])
                y_align = skimage.measure.profile_line(
                    image[img_slice, :, :, align_channel], src, dst, 10, mode="constant"
                )

                p = np.poly1d(
                    np.polyfit(np.arange(0, len(y_align)) * pixelsize, y_align, 10)
                )
                max_number = 50
                t = np.linspace(
                    0, max(np.arange(0, len(y_align)) * pixelsize), max_number
                )

                # get highest peak
                peaks, heights = signal.find_peaks(p(t), max(y_align) * 0.6)
                heights = heights["peak_heights"].tolist()
                # biggest_peak = heights.index(max(heights))
                y_align = y_align.tolist()
                # offset = (peaks[biggest_peak]/max_number)*max(np.arange(0,len(y_align))*pixelsize)
                # Hack:
                biggest_peak = y_align.index(max(y_align))

                max_number = len(y_align)

                ##HACK
                dna = skimage.measure.profile_line(
                    image[img_slice, :, :, 2], src, dst, 1, mode="constant"
                )
                p = np.poly1d(np.polyfit(np.arange(0, len(dna)) * pixelsize, dna, 10))
                max_number = 10000
                # max_number=len(y_align)*1000
                t = np.linspace(0, max(np.arange(0, len(dna)) * pixelsize), max_number)
                # print(len(t))
                yy = (p(t) - min(p(t))) / (max(p(t)) - min(p(t)))
                ##print(yy)

                # offset=biggest_peak
                # print("a")
                pt = yy.tolist()
                closest = pt.index(find_nearest(yy, max(yy) / 2))
                offset = t[closest]
                # def func1(u):
                #    return ((p(u)-min(p(u)))/(max(p(u))-min(p(u))))-(1/2)
                if channel == align_channel:
                    channel_max.append((t[closest] - offset) * scaling)
                    # plt.plot(t[closest]-offset, 0.5, marker='o', markersize=3, color="red")

                if channel != align_channel:
                    y3 = skimage.measure.profile_line(
                        image[img_slice, :, :, channel], src, dst, 10, mode="constant"
                    )
                    p = np.poly1d(np.polyfit(np.arange(0, len(y3)) * pixelsize, y3, 10))
                    max_number = len(y3)
                    t = np.linspace(
                        0, max(np.arange(0, len(y3)) * pixelsize), max_number
                    )
                    # axs.plot((t-offset),(y3-min(y3))/(max(y3)-min(y3)),color = "red")
                    # get highest peak
                    peaks, heights = signal.find_peaks(p(t), max(p(t)) * 0.6)
                    # print(peaks)
                    # print(heights)

                    heights = heights["peak_heights"].tolist()
                    try:
                        biggest_peak2 = heights.index(max(heights))
                        peak_point = peaks[biggest_peak2]
                    except:
                        peak_point = float("NaN")
                    #biggest_peak2 = heights.index(max(heights))
                    peak_point = peaks[biggest_peak2]
                    # print(peak_point)
                    # plt.plot(peak_point-offset, 1, marker='o', markersize=3, color="red")

                # print(func(29))
                # print(f"clo: {closest}")
                # print(t[closest]-offset*pixelsize)
                # x = fsolve(func1,closest)
                # print(f'root: {x} - {func1(x[0])}')

                # axs.plot((t-offset),yy,color = "red")

                # if channel == align_channel:
                #     channel_max.append((biggest_peak - offset) * scaling)
                if channel != align_channel:
                    channel_max.append((peak_point - offset) * scaling)
                    # plt.plot(peak_point-offset, 1, marker='o', markersize=3, color="red")

                ##end_HACK

                # offset = biggest_peak
                # measure:
                y = skimage.measure.profile_line(
                    image[img_slice, :, :, channel], src, dst, 10, mode="constant"
                )
                if normalize == True:
                    if align == True:
                        axs.plot(
                            (np.arange(0, len(y)) - offset) * pixelsize * scaling,
                            (y - min(y)) / (max(y) - min(y)),
                            color=newcolor,
                        )
                    else:
                        axs.plot(
                            (np.arange(0, len(y))) * pixelsize,
                            (y - min(y)) / (max(y) - min(y)),
                            color=newcolor,
                        )
                else:
                    axs.plot(np.arange(0, len(y)) * pixelsize, y, color=newcolor)
            image_peaks[channel].extend(channel_max)

    # print(image_peaks)
    df = pd.DataFrame(image_peaks)
    df = df.transpose()
    df.columns = channels
    # print(df)
    fig, axs = plt.subplots(1)
    sns.swarmplot(data=df)
    fig, axs = plt.subplots(1)
    sns.boxplot(data=df)
    return df

def linescan_half_alingne_lagacy_2c_dnafirst(
    image_path,
    roi_path,
    channels,
    number_of_channels,
    align_channel,
    normalize=True,
    pixelsize=1,
    align=True,
):
    # get roi and image
    scaling = 0.03525845591290619

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def find_first_half(b):
        half_lim = max(b) / 2
        for i, y in enumerate(b):
            if y > half_lim:
                break
        return i

    # create plot canvas
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    image_peaks = [[], []]
    for single_image, single_roi in zip(image_path, roi_path):
        roi = read_roi_zip(single_roi)
        image = io.imread(single_image)
        # print(image.shape)
        # plt.imshow(image[10,0,:,:],cmap="gray")

        # generate a color map iterator
        cmap = plt.get_cmap("tab10")
        colors = iter(cmap.colors)

        for channel in range(number_of_channels):
            newcolor = next(colors)
            # print(newcolor)
            channel_max = []
            # scaling=1
            for key, item in roi.items():
                img_slice = item["position"]["slice"]
                src = (item["y1"], item["x1"])
                dst = (item["y2"], item["x2"])
                y_align = skimage.measure.profile_line(
                    image[img_slice, align_channel, :, :], src, dst, 5, mode="constant"
                )

                p = np.poly1d(
                    np.polyfit(np.arange(0, len(y_align)) * pixelsize, y_align, 10)
                )
                max_number = 50
                t = np.linspace(
                    0, max(np.arange(0, len(y_align)) * pixelsize), max_number
                )

                # get highest peak
                peaks, heights = signal.find_peaks(p(t), max(y_align) * 0.6)
                heights = heights["peak_heights"].tolist()
                # biggest_peak = heights.index(max(heights))
                y_align = y_align.tolist()
                # offset = (peaks[biggest_peak]/max_number)*max(np.arange(0,len(y_align))*pixelsize)
                # Hack:
                biggest_peak = y_align.index(max(y_align))

                max_number = len(y_align)

                ##HACK
                dna = skimage.measure.profile_line(
                    image[img_slice, 1, :, :], src, dst, 5, mode="constant"
                )
                p = np.poly1d(np.polyfit(np.arange(0, len(dna)) * pixelsize, dna, 10))
                max_number = 10000
                # max_number=len(y_align)*1000
                t = np.linspace(0, max(np.arange(0, len(dna)) * pixelsize), max_number)
                # print(len(t))
                yy = (p(t) - min(p(t))) / (max(p(t)) - min(p(t)))
                ##print(yy)

                # offset=biggest_peak
                # print("a")
                pt = yy.tolist()
                # closest = pt.index(find_nearest(yy,max(yy)/2))
                closest = find_first_half(yy)
                offset = t[closest]
                # def func1(u):
                #    return ((p(u)-min(p(u)))/(max(p(u))-min(p(u))))-(1/2)
                if channel == align_channel:
                    channel_max.append((t[closest] - offset) * scaling)
                    # plt.plot(t[closest]-offset, 0.5, marker='o', markersize=3, color="red")

                if channel != align_channel:
                    y3 = skimage.measure.profile_line(
                        image[img_slice, channel, :, :], src, dst, 10, mode="constant"
                    )
                    p = np.poly1d(np.polyfit(np.arange(0, len(y3)) * pixelsize, y3, 10))
                    max_number = len(y3)
                    t = np.linspace(
                        0, max(np.arange(0, len(y3)) * pixelsize), max_number
                    )
                    # axs.plot((t-offset),(y3-min(y3))/(max(y3)-min(y3)),color = "red")
                    # get highest peak
                    peaks, heights = signal.find_peaks(p(t), max(p(t)) * 0.6)
                    # print(peaks)
                    # print(heights)

                    heights = heights["peak_heights"].tolist()
                    try:
                        biggest_peak2 = heights.index(max(heights))
                        peak_point = peaks[biggest_peak2]
                    except:
                        peak_point = float("NaN")
                    #peak_point = peaks[biggest_peak2]
                    channel_max.append((peak_point - offset) * scaling)
                    # print(peak_point)
                    # plt.plot(peak_point-offset, 1, marker='o', markersize=3, color="red")

                # print(func(29))
                # print(f"clo: {closest}")
                # print(t[closest]-offset*pixelsize)
                # x = fsolve(func1,closest)
                # print(f'root: {x} - {func1(x[0])}')

                # axs.plot((t-offset),yy,color = "red")

                # if (channel == align_channel):
                #     channel_max.append((biggest_peak - offset) * scaling)
                # if (channel != align_channel and channel != 0):
                #     channel_max.append((peak_point-offset) * scaling)
                # plt.plot(peak_point-offset, 1, marker='o', markersize=3, color="red")

                ##end_HACK

                # offset = biggest_peak
                # measure:
                y = skimage.measure.profile_line(
                    image[img_slice, channel, :, :], src, dst, 10, mode="constant"
                )
                if normalize == True:
                    if align == True:
                        axs.plot(
                            (np.arange(0, len(y)) - offset) * pixelsize * scaling,
                            (y - min(y)) / (max(y) - min(y)),
                            color=newcolor,
                        )
                        plt.hlines(0.5, -2, 2)
                    else:
                        axs.plot(
                            (np.arange(0, len(y))) * pixelsize,
                            (y - min(y)) / (max(y) - min(y)),
                            color=newcolor,
                        )
                else:
                    axs.plot(np.arange(0, len(y)) * pixelsize, y, color=newcolor)
            image_peaks[channel].extend(channel_max)

    # print(image_peaks)
    df = pd.DataFrame(image_peaks)
    df = df.transpose()
    df.columns = channels
    # print(df)
    fig, axs = plt.subplots(1)
    if align_channel == 0:
        sns.swarmplot(data=df.iloc[:, ::-1])
    else:
        sns.swarmplot(data=df)
    fig, axs = plt.subplots(1)
    if align_channel == 0:
        sns.swarmplot(data=df.iloc[:, ::-1])
    else:
        sns.swarmplot(data=df)
    return df

# def linescan_half_alingne_lagacy_2c_dnafirst(
#     image_path,
#     roi_path,
#     channels,
#     number_of_channels=2,
#     normalize=True,
#     pixelsize=1,
#     align_channel=1,
#     align=True,
# ):
#     # get roi and image
#     scaling = 0.03525845591290619

#     def find_nearest(array, value):
#         array = np.asarray(array)
#         idx = (np.abs(array - value)).argmin()
#         return array[idx]

#     def find_first_half(b):
#         half_lim = max(b) / 2
#         for i, y in enumerate(b):
#             if y > half_lim:
#                 break
#         return i

#     # create plot canvas
#     fig, axs = plt.subplots(1, 1, figsize=(10, 5))
#     image_peaks = [[], []]
#     for single_image, single_roi in zip(image_path, roi_path):
#         roi = read_roi_zip(single_roi)
#         image = io.imread(single_image)
#         print(image.shape)
#         # plt.imshow(image[10,0,:,:],cmap="gray")

#         # generate a color map iterator
#         cmap = plt.get_cmap("tab10")
#         colors = iter(cmap.colors)

#         for channel in range(number_of_channels):
#             newcolor = next(colors)
#             # print(newcolor)
#             channel_max = []
#             # scaling=1
#             for key, item in roi.items():
#                 img_slice = item["position"]["slice"]
#                 src = (item["y1"], item["x1"])
#                 dst = (item["y2"], item["x2"])
#                 y_align = skimage.measure.profile_line(
#                     image[img_slice, align_channel, :, :], src, dst, 10, mode="constant"
#                 )

#                 p = np.poly1d(
#                     np.polyfit(np.arange(0, len(y_align)) * pixelsize, y_align, 10)
#                 )
#                 max_number = 50
#                 t = np.linspace(
#                     0, max(np.arange(0, len(y_align)) * pixelsize), max_number
#                 )

#                 # get highest peak
#                 peaks, heights = signal.find_peaks(p(t), max(y_align) * 0.6)
#                 heights = heights["peak_heights"].tolist()
#                 # biggest_peak = heights.index(max(heights))
#                 y_align = y_align.tolist()
#                 # offset = (peaks[biggest_peak]/max_number)*max(np.arange(0,len(y_align))*pixelsize)
#                 # Hack:
#                 biggest_peak = y_align.index(max(y_align))

#                 max_number = len(y_align)

#                 ##HACK
#                 dna = skimage.measure.profile_line(
#                     image[img_slice, 1, :, :], src, dst, 1, mode="constant"
#                 )
#                 p = np.poly1d(np.polyfit(np.arange(0, len(dna)) * pixelsize, dna, 10))
#                 max_number = 10000
#                 # max_number=len(y_align)*1000
#                 t = np.linspace(0, max(np.arange(0, len(dna)) * pixelsize), max_number)
#                 # print(len(t))
#                 yy = (p(t) - min(p(t))) / (max(p(t)) - min(p(t)))
#                 ##print(yy)

#                 # offset=biggest_peak
#                 # print("a")
#                 pt = yy.tolist()
#                 # closest = pt.index(find_nearest(yy,max(yy)/2))
#                 closest = find_first_half(yy)
#                 offset = t[closest]
#                 # def func1(u):
#                 #    return ((p(u)-min(p(u)))/(max(p(u))-min(p(u))))-(1/2)
#                 if channel == 1:
#                     channel_max.append((t[closest] - offset) * scaling)
#                     # plt.plot(t[closest]-offset, 0.5, marker='o', markersize=3, color="red")

#                 if channel != align_channel:
#                     y3 = skimage.measure.profile_line(
#                         image[img_slice, channel, :, :], src, dst, 10, mode="constant"
#                     )
#                     p = np.poly1d(np.polyfit(np.arange(0, len(y3)) * pixelsize, y3, 10))
#                     max_number = len(y3)
#                     t = np.linspace(
#                         0, max(np.arange(0, len(y3)) * pixelsize), max_number
#                     )
#                     # axs.plot((t-offset),(y3-min(y3))/(max(y3)-min(y3)),color = "red")
#                     # get highest peak
#                     peaks, heights = signal.find_peaks(p(t), max(p(t)) * 0.6)
#                     # print(peaks)
#                     # print(heights)

#                     heights = heights["peak_heights"].tolist()
#                     biggest_peak2 = heights.index(max(heights))
#                     peak_point = peaks[biggest_peak2]
#                     channel_max.append((peak_point - offset) * scaling)
#                     # print(peak_point)
#                     # plt.plot(peak_point-offset, 1, marker='o', markersize=3, color="red")

#                 # print(func(29))
#                 # print(f"clo: {closest}")
#                 # print(t[closest]-offset*pixelsize)
#                 # x = fsolve(func1,closest)
#                 # print(f'root: {x} - {func1(x[0])}')

#                 # axs.plot((t-offset),yy,color = "red")

#                 # if (channel == align_channel):
#                 #     channel_max.append((biggest_peak - offset) * scaling)
#                 # if (channel != align_channel and channel != 0):
#                 #     channel_max.append((peak_point-offset) * scaling)
#                 # plt.plot(peak_point-offset, 1, marker='o', markersize=3, color="red")

#                 ##end_HACK

#                 # offset = biggest_peak
#                 # measure:
#                 y = skimage.measure.profile_line(
#                     image[img_slice, channel, :, :], src, dst, 10, mode="constant"
#                 )
#                 if normalize == True:
#                     if align == True:
#                         axs.plot(
#                             (np.arange(0, len(y)) - offset) * pixelsize * scaling,
#                             (y - min(y)) / (max(y) - min(y)),
#                             color=newcolor,
#                         )
#                         plt.hlines(0.5, -2, 2)
#                     else:
#                         axs.plot(
#                             (np.arange(0, len(y))) * pixelsize,
#                             (y - min(y)) / (max(y) - min(y)),
#                             color=newcolor,
#                         )
#                 else:
#                     axs.plot(np.arange(0, len(y)) * pixelsize, y, color=newcolor)
#             image_peaks[channel].extend(channel_max)

#     # print(image_peaks)
#     df = pd.DataFrame(image_peaks)
#     df = df.transpose()
#     df.columns = channels
#     # print(df)
#     fig, axs = plt.subplots(1)
#     sns.swarmplot(data=df)
#     fig, axs = plt.subplots(1)
#     sns.boxplot(data=df)
#     return df
