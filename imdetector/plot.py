import glob
import os
import shutil
import time
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

graybox = {
    "facecolor": "#DDDDDD",
    "edgecolor": "#DDDDDD",
    "boxstyle": "Round",
    "linewidth": 10
}
graybutton = {
    "s": "Inapplicable",
    "color": "white",
    "fontweight": "bold",
    "bbox": graybox,
    "fontsize": 14,
}

greenbox = {
    "facecolor": "#18BC9C",
    "edgecolor": "#18BC9C",
    "boxstyle": "Round",
    "linewidth": 10
}
safebutton = {
    "s": "Not Detected",
    "color": "white",
    "fontweight": "bold",
    "bbox": greenbox,
    "fontsize": 14,
}

redbox = {
    "facecolor": "#E74C3C",
    "edgecolor": "#E74C3C",
    "boxstyle": "Round",
    "linewidth": 10
}
alertbutton = {
    "s": "Suspicious",
    "color": "white",
    "fontweight": "bold",
    "bbox": redbox,
    "fontsize": 14,
}

buttondic = {
    1: alertbutton,
    0: safebutton,
    -1: graybutton
}


def plot_pdfpages(suspicious_images, result_imgnames, result_imgarrs, result_ratios, date, SAVE_DIR):
    pp = PdfPages(os.path.join(
        SAVE_DIR,
        '{}.pdf'.format(date.strftime('%Y-%m-%d-%H%M%S'))))
    len_sus = len(suspicious_images)
    title = date.strftime('%Y/%m/%d %H:%M:%S')
    for i, img in enumerate(suspicious_images):
        plot_report(img, len_sus, title, i, pp, SAVE_DIR)
    plot_duplication_report(result_imgnames, result_imgarrs,
                            result_ratios, title, pp, SAVE_DIR)
    pp.close()


def plot_report(img, len_sus, title, i, pp, SAVE_DIR):
    H, W = img.gray.shape

    fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    # fig.suptitle('{}  No. {}'.format(title, i),
    #              fontweight='bold', fontsize=12, y=1.02)

    # Input image #
    ax = fig.add_subplot(521)
    size = img.size / 1000
    ax.text(0, 0.95, title)
    ax.text(0, 0.75, 'No. {}'.format(
        i), fontsize=20, fontweight='bold')
    # ax.text(0, 0.5, 'File name: {}'.format(img.name))
    ax.text(0, 0.55, 'Image size: {} x {} pixels'.format(H, W))
    ax.text(0, 0.4, 'File size: {:.3g} KB'.format(int(size)))
    if size < 10:
        ax.text(
            0,
            0.1,
            'Too small file size.\nProbably poor image quality.',
            color="red")
    ax.axis('off')
    # ax.set_title('Pixel value histogram')
    # ax.hist(img.gray.flatten(), bins=256, range=(0, 256))
    ax = fig.add_subplot(522)
    ax.imshow(cv.cvtColor(img.mat, cv.COLOR_BGR2RGB), aspect="equal")
    ax.axis('off')

    # Cut-Paste (histogram equalization) #
    ax = fig.add_subplot(523)
    ax.text(0, 0.75, 'Histogram equalization', fontsize=18, fontweight='bold')
    ax.axis('off')
    ax = fig.add_subplot(524)
    # ax.set_title('Histogram equalization')
    ax.imshow(img.keyimg[img.gap:-img.gap, img.gap:-img.gap])
    ax.axis('off')

    # Noise #
    ax = fig.add_subplot(525)
    ax.text(0, 0.75, 'Noise', fontsize=18, fontweight='bold')
    if img.noise == 1:
        ax.text(
            0,
            0.5,
            'Noisy',
            fontweight='bold',
            fontsize=14,
            color="white",
            bbox=redbox)
        ax.text(
            0.3,
            0.5,
            'Following detection may be less correct.',
            color='red',
            fontsize=10,
        )
    ax.axis('off')
    ax = fig.add_subplot(526)
    ax.imshow(img.no_img)
    ax.axis('off')

    # Clipping #
    ax = fig.add_subplot(527)
    ax.text(0, 0.75, 'Clipping', fontsize=18, fontweight='bold')
    button = buttondic[img.clipping]
    ax.text(0., 0.5, **button)
    ax.text(0, 0.25, 'Area ratio: {} %'.format(int(img.area_ratio * 100)))
    ax.axis('off')
    ax = fig.add_subplot(528)
    im = img.cl_img if hasattr(img, 'cl_img') else img.mat
    ax.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    ax.axis('off')

    # Copy-Move #
    ax = fig.add_subplot(529)
    ax.text(0, 0.75, 'Copy-move', fontsize=18, fontweight='bold')
    button = buttondic[img.copymove]
    ax.text(0, 0.5, **button)
    if img.copymove is -1:
        ax.text(0, 0.25, 'Not enough keypoints were extracted.', color='red')
    else:
        ax.text(0, 0.25, 'Inlier keypoint ratio: {} %'.format(
            int(img.mask_ratio * 100)))
    ax.axis('off')
    ax = fig.add_subplot(5, 2, 10)
    im = img.cm_img if hasattr(img, 'cm_img') else img.mat
    ax.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    ax.axis('off')

    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(
        pp,
        # bbox_inches='tight',
        dpi=fig.dpi,
        format='pdf')
    fig.clf()


def plot_row(fig, result_imgnames, result_imgarrs, result_ratios, i, j):
    ax = fig.add_subplot(4, 2, 2 * i + 1)
    ax.text(0, 0.75, **buttondic[1])
    ax.text(0, 0.45, 'Left: No. {}\nRight: No. {}'.format(
        result_imgnames[j][0], result_imgnames[j][1]))
    ax.text(0, 0.25, 'Inlier keypoint ratio: {} %'.format(
        int(result_ratios[j] * 100)))
    ax.axis('off')
    ax = fig.add_subplot(4, 2, 2 * i + 2)
    ax.imshow(cv.cvtColor(result_imgarrs[j], cv.COLOR_BGR2RGB))
    ax.axis('off')


def plot_duplication_report(
        result_imgnames,
        result_imgarrs,
        result_ratios,
        title,
        pp,
        SAVE_DIR):
    len_img = len(result_ratios)
    page = (len_img - 1) // 4
    for p in range(page + 1):
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        fig.suptitle('{}  Duplication check {} of {}'.format(
            title, p + 1, page + 1), fontweight='bold', fontsize=12, y=0.98)

        for i in range(4):
            j = p * 4 + i
            if j < len_img:
                plot_row(
                    fig,
                    result_imgnames,
                    result_imgarrs,
                    result_ratios,
                    i,
                    j)
            else:
                ax = fig.add_subplot(4, 2, 2 * i + 1)
                ax.axis('off')
                ax = fig.add_subplot(4, 2, 2 * i + 2)
                ax.axis('off')

        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(
            pp,
            # bbox_inches='tight',
            dpi=fig.dpi,
            format='pdf')
        fig.clf()
