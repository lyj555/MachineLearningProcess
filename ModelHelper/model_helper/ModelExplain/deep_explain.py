# -*- coding: utf-8 -*-

import os
os.environ['KERAS_BACKEND'] = "tensorflow"

from datetime import datetime
from collections import Iterable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from keras.models import Model
from keras.layers import Lambda

import shap


colors = []
for l in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255, l))
for l in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255, l))
SHAP_COLOR_MAP = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


def _get_shap_matrix(img, model, ith_predict, background_sample, output_dimension):
    if len(output_dimension) == 2:
        out = Lambda(lambda x, index: x[:, index, :], arguments={"index": ith_predict})(model.layers[-1].output)
        model_temp = Model(inputs=model.input, outputs=out)

        explainer = shap.DeepExplainer(model_temp, background_sample)
        shap_values, indexs = explainer.shap_values(img, ranked_outputs=1)  # ranked_output*1*img_shape
    elif len(output_dimension) == 1:
        explainer = shap.DeepExplainer(model, background_sample)
        shap_values, indexs = explainer.shap_values(img, ranked_outputs=1)
    else:
        raise ValueError("the model's output dimension must be 1 or 2!")
    return shap_values, indexs


def explain_image_plot(images, model, background_sample, filter_blank=False, blank_index_list=None, index_map=None,
                       row_scale=3, col_scale=3, fontsize=10, back_alpha=0.15, hspace=0.2, show=True):
    """
    explain model images
    :param images: numpy.array, (batch_size, ...)
    :param model: keras model,
    :param background_sample: numpy.array, (batch_size, ...), suggest randomly select 100 samples, the more the samples,
    the lower the shap computation speed, the more precise shap plot.
    :param filter_blank: bool, set False, means all predict position will be explained, set True, omit some position
    :param blank_index_list: Iterable, valid only filter_blank = True
    :param index_map: dict or None, index to label, set None use the default index
    :param row_scale: int, image row scale, default 3
    :param col_scale: int, image col scale, default 3
    :param fontsize: int, the fontsize of every subplot image, default 10
    :param back_alpha: float, the raw image display scale, default 0.15
    :param hspace: float, the subplots's gap, default 0.2
    :param show: bool, if show the plot
    :return: tuple, fig, axes
    """
    assert isinstance(model, Model), "input model must be keras like model!"
    output_dimension = model.layers[-1].output.shape[1:]  # exclude the batch_size dimension
    assert len(output_dimension) <= 2, "the model's output dimension must be less than 2!"
    if len(output_dimension) == 2:  # self-define explain plot, default shap can not process this.
        max_sequence_length, possible_tag_num = tuple(map(lambda x: x.value, output_dimension))

        if filter_blank:
            assert isinstance(blank_index_list, Iterable), "blank_index_list must be a Iterable object!"
            preds = model.predict(images)  # len(image) * max_sequence_length * possible_tag_num
            indexs = []  # the needed index in every image
            for i in range(len(images)):
                y_predict = preds[i].argmax(axis=1)
                _index = []
                for index, val in enumerate(y_predict):
                    if val not in blank_index_list:
                        _index.append(index)
                indexs.append(_index)
        else:
            indexs = []
            for i in range(len(images)):
                indexs.append(range(max_sequence_length))
    else:
        indexs = [(0, ) for _ in range(len(images))]

    row_num = len(images)
    col_num = max(map(lambda x: len(x), indexs)) + 1  # +1 used to plot raw image

    fig_size = (col_scale * col_num, row_scale * row_num)
    fig, axes = plt.subplots(nrows=row_num, ncols=col_num, figsize=fig_size)

    if len(axes.shape) == 1:
        axes = axes.reshape(1, axes.size)

    for row_index in range(row_num):
        t1 = datetime.now()
        print(f"{row_index+1}/{row_num} is processing...")
        x_curr = images[row_index].copy().astype("float")  # current image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])
        if x_curr.max() > 1:
            x_curr /= 255.

        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = (0.2989 * x_curr[:, :, 0] + 0.5870 * x_curr[:, :, 1] + 0.1140 * x_curr[:, :, 2])  # rgb to gray
        else:
            x_curr_gray = x_curr

        axes[row_index, 0].imshow(x_curr, cmap=plt.get_cmap('gray'))
        axes[row_index, 0].axis('off')
        axes[row_index, 0].set_title("raw image gray style", fontsize=fontsize)

        need_plot_index = indexs[row_index]
        for col_index in range(1, col_num):
            print(f"\t{col_index}/{col_num-1} is plotting...")
            try:
                plot_index = need_plot_index[col_index-1]
            except IndexError:
                axes[row_index, col_index].axis("off")
                continue
            shap_values, max_indexs = _get_shap_matrix(img=np.array([images[row_index]]), model=model,
                                                       ith_predict=plot_index, background_sample=background_sample,
                                                       output_dimension=output_dimension)
            if len(shap_values[0][0].shape) == 2:
                abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
            else:
                abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()
            max_val = np.nanpercentile(abs_vals, 99.9)

            if index_map is not None:
                index_names = np.vectorize(lambda x: index_map[x])(max_indexs)[0]
            else:
                index_names = max_indexs[0]

            sv = shap_values[0][0] if len(shap_values[0][0].shape) == 2 else shap_values[0][0].sum(-1)
            # first plot raw image gray style
            axes[row_index, col_index].imshow(x_curr_gray, cmap=plt.get_cmap("gray"), alpha=back_alpha,
                                              extent=(-1, sv.shape[1], sv.shape[0], -1))
            # second plot the shap value
            if row_index == row_num - 1 and col_index == col_num - 1:
                image_colorbar = axes[row_index, col_index].imshow(sv, cmap=SHAP_COLOR_MAP, vmin=-max_val, vmax=max_val)
            else:
                axes[row_index, col_index].imshow(sv, cmap=SHAP_COLOR_MAP, vmin=-max_val, vmax=max_val)
            axes[row_index, col_index].axis("off")

            axes[row_index, col_index].set_title(f"{index_names[0]}", fontsize=fontsize)
        t2 = datetime.now()
        print(f"{row_index+1}/{row_num} done, cost {(t2-t1).seconds}")
    if hspace == "auto":
        fig.tight_layout()
    else:
        #         fig.subplots_adjust(hspace=hspace)
        fig.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, hspace=hspace)

    cb = fig.colorbar(image_colorbar, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal",
                      fraction=0.05, aspect=30)  # fraction control distance axes and colorbar, aspect control the width of colorbar
    cb.outline.set_visible(False)
    if show:
        plt.show()
    return fig, axes


# if __name__ == "__main__":
#     import os
#     from PIL import Image
#     import cv2
#     from keras.models import load_model
#
#     def process_image(file_path, img_width=270, img_height=40):
#         X = []
#         img = Image.open(file_path).convert('L')
#         img = np.array(img)
#
#         img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
#         img = cv2.transpose(img, (img_height, img_width))
#         img = cv2.flip(img, 1)
#         img = (255 - img) / 256  # 反色处理
#         X.append([img])
#
#         X = np.transpose(X, (0, 2, 3, 1))
#         return np.array(X)
#
#
#     def create_backgroud_sample(file_dir, pre_process, bac_num=50):
#         pictures = []
#         for i in os.listdir(file_dir)[:bac_num]:
#             file_path = os.path.join(file_dir, i)
#             pictures.append(pre_process(file_path)[0])
#         return np.array(pictures)
#
#     model_path = "./model/ctcbig.hdf5base"
#     back_group_sample_path = "./backgroud_sample"
#     explain_sample_path = "./needed_explain_sample"
#
#     model = load_model(model_path)
#     bc = create_backgroud_sample(back_group_sample_path, pre_process=process_image, bac_num=50)
#     print("backgroup sample shape", bc.shape)
#
#     # explain_one_example_path = "./needed_explain_sample/021-58358378.jpg"
#     # images = process_image(file_path=explain_one_example_path)
#     images = []
#     for i in os.listdir(explain_sample_path)[:3]:
#         image_path = os.path.join(explain_sample_path, i)
#         image = process_image(file_path=image_path)[0]
#         images.append(image)
#     images = np.array(images)
#     print("images shape", images.shape)
#
#     index_map = dict(zip(range(10), range(10)))
#     index_map[10] = "-"
#     index_map[11] = "blank"
#
#     fig, axes = explain_image_plot(images=images, model=model, background_sample=bc, filter_blank=True,
#                                    blank_index_list=(11, ), index_map=index_map, show=False)
#     fig.savefig("./test_images.png")
#
