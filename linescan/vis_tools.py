"""Contains all of the visualization tools for the line_scan application"""

from read_roi import read_roi_zip
import matplotlib.pyplot as plt
from skimage import io, measure


def plot_image_and_profiles(
    image_path, roi_path, disp_channels=None, number_of_channels=2
):
    """Plots every z-slice that has at least one linescan.
    Colors represent the particular line drawn on the image.
    Channels are displayed one after each other.
    Channels can be normalized and the fitting and peak calling can be displayed.
    Further lines can be aligned according to the offset of peak in a praticular channel.
    """
    if number_of_channels == 2:
        plot_image_and_profiles_2c(image_path, roi_path, disp_channels=disp_channels)
    elif number_of_channels == 3:
        plot_image_and_profiles_3c(image_path, roi_path, disp_channels=disp_channels)
    else:
        print(
            f"The channel number {number_of_channels}, is not supported in this version"
        )


# TODO: Add 3,4 channel + Error
# TODO: scaling
# TODO: label slice in images
# TODO: allow scaling and
# TODO: scale along axis of image in microns


def plot_image_and_profiles_2c(image_path, roi_path, disp_channels=None):
    """The two channel implementation of plot_image_and_profiles."""
    # get roi
    if disp_channels is None:
        disp_channels = [0, 1]
    if len(disp_channels) > 3:
        raise ValueError("You are trying to display more channels then your image has!")
    number_of_channels = 2
    roi = read_roi_zip(roi_path)
    # get slice set
    slice_set = set()
    for _, item in roi.items():
        slice_set.add(item["position"]["slice"])
    slice_set = sorted(slice_set)

    _, axs = plt.subplots(
        len(slice_set),
        number_of_channels + len(disp_channels),
        figsize=(15, 5 * len(slice_set)),
    )
    image = io.imread(image_path)
    for slice_num, img_slice in enumerate(slice_set):
        # Plot linescans of all channels
        for channel in range(number_of_channels):
            for _, item in roi.items():
                if item["position"]["slice"] == img_slice:
                    src = (item["y1"], item["x1"])
                    dst = (item["y2"], item["x2"])
                    values = measure.profile_line(
                        image[img_slice, channel, :, :], src, dst, 1, mode="constant"
                    )
                    axs[slice_num, channel].plot(range(len(values)), values)
        # Draw images with lines for the set of channels to display
        for disp_num, disp_channel in enumerate(disp_channels):
            axs[slice_num, number_of_channels + disp_num].imshow(
                image[img_slice, disp_channel, :, :], cmap="gray"
            )
            for _, item in roi.items():
                if item["position"]["slice"] == img_slice:
                    x_values = [item["x1"], item["x2"]]
                    y_values = [item["y1"], item["y2"]]
                    axs[slice_num, number_of_channels + disp_num].plot(
                        x_values, y_values
                    )


def plot_image_and_profiles_3c(
    image_path, roi_path, disp_channels=[0], number_of_channels=3
):
    # get roi
    roi = read_roi_zip(roi_path)
    # get slice set
    slice_set = set()
    for key, item in roi.items():
        slice_set.add(item["position"]["slice"])
    slice_set = sorted(slice_set)

    fig, axs = plt.subplots(
        len(slice_set),
        number_of_channels + len(disp_channels),
        figsize=(15, 5 * len(slice_set)),
    )
    image = io.imread(image_path)
    for round, img_slice in enumerate(slice_set):
        # Plot linescans of all channels
        for channel in range(number_of_channels):
            for key, item in roi.items():
                if item["position"]["slice"] == img_slice:
                    src = (item["y1"], item["x1"])
                    dst = (item["y2"], item["x2"])
                    y = measure.profile_line(
                        image[img_slice, :, :, channel],
                        src,
                        dst,
                        1,
                        mode="constant"
                        # TODO merge with 2c since this is the only difference
                    )
                    axs[round, channel].plot(range(len(y)), y)
        # Draw images with lines for the set of channels to display
        for disp_num, disp_channel in enumerate(disp_channels):
            axs[round, number_of_channels + disp_num].imshow(
                image[img_slice, :, :, disp_channel], cmap="gray"
            )
            for key, item in roi.items():
                if item["position"]["slice"] == img_slice:
                    x_values = [item["x1"], item["x2"]]
                    y_values = [item["y1"], item["y2"]]
                    axs[round, number_of_channels + disp_num].plot(x_values, y_values)


# TODO: Also do not draw lineprofile if not displayed
def plot_line_profiles(image_path, roi_path, disp_channels=None, number_of_channels=3):
    """Plots every every linescan together with the image slice it was drawn on.
    Colors represent the individual channels, the line drawn is diplayed on top of the image.
    The line profiles can be normalized, and the fitting and peak calling can be displayed.
    Values are in pixel unless the user provides the pixel size in microns."""
    if number_of_channels == 2 or number_of_channels == 3:
        plot_line_profiles_2c(
            image_path,
            roi_path,
            disp_channels=disp_channels,
            number_of_channels=number_of_channels,
        )
    else:
        print(
            f"The channel number {number_of_channels}, is not supported in this version"
        )


def plot_line_profiles_2c(
    image_path, roi_path, disp_channels=None, number_of_channels=2
):
    """The two channel implementation for the plot_line_profiles"""
    # get roi
    if disp_channels is None:
        disp_channels = [0, 1]
    if len(disp_channels) > number_of_channels:
        raise ValueError("You are trying to display more channels then your image has!")

    roi = read_roi_zip(roi_path)
    # get slice set
    # TODO: count simpler
    slice_count = 0
    for _, item in roi.items():
        slice_count += 1

    _, axs = plt.subplots(
        slice_count, 1 + len(disp_channels), figsize=(15, 5 * slice_count)
    )
    image = io.imread(image_path)

    for item_num, item in enumerate(roi.items()):
        _, item = item
        img_slice = item["position"]["slice"]
        src = (item["y1"], item["x1"])
        dst = (item["y2"], item["x2"])
        cmap = plt.get_cmap("tab10")
        colors = iter(cmap.colors)
        # TODO: cleaner color management
        for channel in range(number_of_channels):
            new_color = next(colors)
            if number_of_channels == 2:
                values = measure.profile_line(
                    image[img_slice, channel, :, :], src, dst, 1, mode="constant"
                )
            if number_of_channels == 3:
                values = measure.profile_line(
                    image[img_slice, :, :, channel], src, dst, 1, mode="constant"
                )
            else:
                raise ValueError(
                    f"Your channel number: {number_of_channels} is not supported."
                )

            axs[item_num, 0].plot(range(len(values)), values, color=new_color)
        cmap = plt.get_cmap("tab10")
        colors = iter(cmap.colors)
        for disp_num, disp_channel in enumerate(disp_channels):
            new_color = next(colors)
            if number_of_channels == 2:
                axs[item_num, 1 + disp_num].imshow(
                    image[img_slice, disp_channel, :, :], cmap="gray"
                )
            if number_of_channels == 3:
                axs[item_num, 1 + disp_num].imshow(
                    image[img_slice, :, :, disp_channel], cmap="gray"
                )
            else:
                raise ValueError(
                    f"Your channel number: {number_of_channels} is not supported."
                )
            x_values = [item["x1"], item["x2"]]
            y_values = [item["y1"], item["y2"]]
            axs[item_num, 1 + disp_num].plot(x_values, y_values, color=new_color)
        # Draw images with lines for the set of channels to display
        cmap = plt.get_cmap("tab10")
        colors = iter(cmap.colors)
        for disp_num, disp_channel in enumerate(disp_channels):
            new_color = next(colors)
            if number_of_channels == 2:
                axs[item_num, 1 + disp_num].imshow(
                    image[img_slice, disp_channel, :, :], cmap="gray"
                )
            if number_of_channels == 3:
                axs[item_num, 1 + disp_num].imshow(
                    image[img_slice, :, :, disp_channel], cmap="gray"
                )
            else:
                raise ValueError(
                    f"Your channel number: {number_of_channels} is not supported."
                )
            x_values = [item["x1"], item["x2"]]
            y_values = [item["y1"], item["y2"]]
            axs[item_num, 1 + disp_num].plot(x_values, y_values, color=new_color)


# def image_and_line(axs,item_num,disp_num,image,item):
#     axs[item_num, 1 + disp_num].imshow(
#         image[img_slice, disp_channel, :, :], cmap="gray"
#     )
#     x_values = [item["x1"], item["x2"]]
#     y_values = [item["y1"], item["y2"]]
#     axs[item_num, 1 + disp_num].plot(x_values, y_values, color=new_color)
