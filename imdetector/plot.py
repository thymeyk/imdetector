import glob
import os
import shutil
import time
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

graybox = {
    "facecolor": "#DDDDDD",
    "edgecolor": "#DDDDDD",
    "boxstyle": "Round",
    "linewidth": 5
}
graybutton = {
    "s": "Inapplicable",
    "color": "white",
    "fontweight": "bold",
    "bbox": graybox,
    "fontsize": 16,
}

greenbox = {
    "facecolor": "#18BC9C",
    "edgecolor": "#18BC9C",
    "boxstyle": "Round",
    "linewidth": 5
}
safebutton = {
    "s": "Not Detected",
    "color": "white",
    "fontweight": "bold",
    "bbox": greenbox,
    "fontsize": 16,
}

redbox = {
    "facecolor": "#E74C3C",
    "edgecolor": "#E74C3C",
    "boxstyle": "Round",
    "linewidth": 5
}
alertbutton = {
    "s": "Suspicious",
    "color": "white",
    "fontweight": "bold",
    "bbox": redbox,
    "fontsize": 16,
}

buttondic = {
    1: alertbutton,
    0: safebutton,
    -1: graybutton
}


def plot_pdfpages(suspicious_images, date, SAVE_DIR):
    pp = PdfPages(os.path.join(
        SAVE_DIR,
        '{}.pdf'.format(date.strftime('%Y-%m-%d-%H%M%S'))))
    len_sus = len(suspicious_images)
    for i, img in enumerate(suspicious_images):
        plot_report(img, len_sus, date.strftime(
            '%Y/%m/%d %H:%M:%S'), i, pp, SAVE_DIR)
    pp.close()


def plot_report(img, len_sus, title, i, pp, SAVE_DIR):
    H, W = img.gray.shape

    fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    fig.suptitle('{}  {} of {}'.format(title, i + 1, len_sus),
                 fontweight='bold', fontsize=12, y=1.02)

    # Input image #
    ax = fig.add_subplot(521)
    ax.set_title('Input image')
    ax.imshow(cv.cvtColor(img.mat, cv.COLOR_BGR2RGB), aspect="equal")
    ax.axis('off')
    ax = fig.add_subplot(522)
    ax.set_title('Pixel value histogram')
    ax.hist(img.gray.flatten(), bins=256, range=(0, 256))

    # Cut-Paste (histogram equalization) #
    ax = fig.add_subplot(523)
    ax.set_title('Histogram equalization')
    ax.imshow(img.keyimg[img.gap:-img.gap, img.gap:-img.gap])
    ax.axis('off')
    ax = fig.add_subplot(524)
    size = img.size / 1000
    ax.text(0, 0.9, 'File name: {}'.format(img.name))
    ax.text(0, 0.7, 'Image size: {} x {} pixels'.format(H, W))
    ax.text(0, 0.5, 'File size: {:.3g} KB'.format(int(size)))
    ax.axis('off')

    # Noise #
    ax = fig.add_subplot(525)
    ax.set_title('Noise')
    ax.imshow(img.no_img)
    ax.axis('off')
    ax = fig.add_subplot(526)
    if img.noise == 1:
        ax.text(
            0,
            0.3,
            'Noisy',
            fontweight='bold',
            fontsize=10,
            color="white",
            bbox=redbox)
        ax.text(
            0.3,
            0.3,
            'Following detection would be less correct.',
            color='red',
            fontsize=10,
        )
    ax.axis('off')

    # Clipping #
    ax = fig.add_subplot(527)
    ax.set_title('Clipping')
    im = img.cl_img if hasattr(img, 'cl_img') else img.mat
    ax.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    ax.axis('off')
    ax = fig.add_subplot(528)
    button = buttondic[img.clipping]
    ax.text(0, 0.8, **button)
    ax.text(0, 0.6, 'Area ratio: {} %'.format(int(img.area_ratio * 100)))
    if size < 10:
        ax.text(
            0,
            0.3,
            'Too small file size.\nProbably poor image quality.',
            color="red")
    ax.axis('off')

    # Copy-Move #
    ax = fig.add_subplot(529)
    ax.set_title('Copy-move')
    im = img.cm_img if hasattr(img, 'cm_img') else img.mat
    ax.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    ax.axis('off')
    ax = fig.add_subplot(5, 2, 10)
    button = buttondic[img.copymove]
    ax.text(0, 0.8, **button)
    if img.copymove is -1:
        ax.text(0, 0.6, 'Not enough keypoints were extracted.', color='red')
    else:
        ax.text(0, 0.6, 'Inlier keypoint ratio: {} %'.format(
            int(img.mask_ratio * 100)))
    ax.axis('off')

    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(
        # os.path.join(SAVE_DIR, '{}_{}.pdf'.format(title, i + 1)),
        pp,
        bbox_inches='tight',
        dpi=fig.dpi,
        format='pdf')
    fig.clf()
