from read_roi import read_roi_zip
import matplotlib.pyplot as plt
from skimage import io, measure


def plot_image_and_profiles(
    image_path, roi_path, disp_channels=None, number_of_channels=2
):
    if number_of_channels == 2:
        plot_image_and_profiles_2c(image_path, roi_path, disp_channels=disp_channels)
    else:
        print(
            f"The channel number {number_of_channels}, is not supported in this version"
        )
#TODO: Add 3,4 channel + Error
#TODO: scaling
#TODO: label slice in images
#TODO: allow scaling and 

def plot_image_and_profiles_2c(image_path, roi_path, disp_channels=None):
    # get roi
    if disp_channels is None:
        disp_channels = [0, 1]
    if len(disp_channels) > 2:
        raise ValueError("You are trying to display more channels then your image has!")
    number_of_channels = 2
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
                        image[img_slice, channel, :, :], src, dst, 1, mode="constant"
                    )
                    axs[round, channel].plot(range(len(y)), y)
        # Draw images with lines for the set of channels to display
        for disp_num, disp_channel in enumerate(disp_channels):
            axs[round, number_of_channels + disp_num].imshow(
                image[img_slice, disp_channel, :, :], cmap="gray"
            )
            for key, item in roi.items():
                if item["position"]["slice"] == img_slice:
                    x_values = [item["x1"], item["x2"]]
                    y_values = [item["y1"], item["y2"]]
                    axs[round, number_of_channels + disp_num].plot(x_values, y_values)


def plot_line_profiles(image_path, roi_path, disp_channels=None, number_of_channels=2):
    if number_of_channels == 2:
        plot_line_profiles_2c(image_path, roi_path, disp_channels=disp_channels)
    else:
        print(
            f"The channel number {number_of_channels}, is not supported in this version"
        )


def plot_line_profiles_2c(
    image_path, roi_path, disp_channels=None, number_of_channels=2
):
    # get roi
    if disp_channels is None:
        disp_channels = [0, 1]
    if len(disp_channels) > 2:
        raise ValueError("You are trying to display more channels then your image has!")
    number_of_channels = 2

    roi = read_roi_zip(roi_path)
    # get slice set
    # TODO: count simpler
    slice_list = list()
    for key, item in roi.items():
        slice_list.append(item["position"]["slice"])
    slice_list = sorted(slice_list)
    # print(len(slice_list))

    fig, axs = plt.subplots(
        len(slice_list), 1 + len(disp_channels), figsize=(15, 5 * len(slice_list))
    )
    image = io.imread(image_path)

    for item_num, item in enumerate(roi.items()):
        key, item = item
        img_slice = item["position"]["slice"]
        src = (item["y1"], item["x1"])
        dst = (item["y2"], item["x2"])
        cmap = plt.get_cmap("tab10")
        colors = iter(cmap.colors)
        # TODO: cleaner color management
        for channel in range(number_of_channels):
            new_color = next(colors)
            y = measure.profile_line(
                image[img_slice, channel, :, :], src, dst, 1, mode="constant"
            )
            axs[item_num, 0].plot(range(len(y)), y, color=new_color)
        cmap = plt.get_cmap("tab10")
        colors = iter(cmap.colors)
        for disp_num, disp_channel in enumerate(disp_channels):
            new_color = next(colors)
            axs[item_num, 1 + disp_num].imshow(
                image[img_slice, disp_channel, :, :], cmap="gray"
            )
            x_values = [item["x1"], item["x2"]]
            y_values = [item["y1"], item["y2"]]
            axs[item_num, 1 + disp_num].plot(x_values, y_values, color=new_color)
        # Draw images with lines for the set of channels to display
        cmap = plt.get_cmap("tab10")
        colors = iter(cmap.colors)
        for disp_num, disp_channel in enumerate(disp_channels):
            new_color = next(colors)
            axs[item_num, 1 + disp_num].imshow(
                image[img_slice, disp_channel, :, :], cmap="gray"
            )
            x_values = [item["x1"], item["x2"]]
            y_values = [item["y1"], item["y2"]]
            axs[item_num, 1 + disp_num].plot(x_values, y_values, color=new_color)
