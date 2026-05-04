import os
import matplotlib.pyplot as plt
from joblib import dump, load
from pyts.image import GramianAngularField
import numpy as np
from PIL import Image
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

def makeTimeFrequencyImage(signal, index, path, img_size=224):
    signal = np.array(signal)

    gaf = GramianAngularField(image_size=1024, method='difference')
    gaf_image = gaf.fit_transform(signal.reshape(1, -1))

    plt.imshow(gaf_image[0], cmap='jet', origin='lower')

    plt.axis('off')
    plt.gcf().set_size_inches(img_size / 100, img_size / 100)
    img_path = path + 'signal_' + str(index) + '.png'
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()


if __name__ == '__main__':
    train_data = load('./dataresult_A/train_data')
    val_data = load('./dataresult_A/val_data')
    test_data = load('./dataresult_A/test_data')

    data_set = [train_data, val_data, test_data]

    train_path = './GADFImages_A/train/'
    val_path = './GADFImages_A/val/'
    test_path = './GADFImages_A/test/'

    path_list = [train_path, val_path, test_path]

    for item in range(len(data_set)):
        dataset = data_set[item]
        path = path_list[item]

        for index, signal in enumerate(dataset):
            makeTimeFrequencyImage(signal[:-1], index, path)
