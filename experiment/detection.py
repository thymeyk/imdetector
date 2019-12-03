import glob
import os
import shutil
import time
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imdetector.image import SuspiciousImage
from imdetector.noise import Noise
from imdetector.clipping import Clipping
from imdetector.copymove import CopyMove, Duplication
from imdetector.cutpaste import CutPaste

graybox = {
    "facecolor": "#DDDDDD",
    "edgecolor": "#DDDDDD",
    "boxstyle": "Round",
    "linewidth": 2
}
graybutton = {
    "s": "Inapplicable",
    "color": "white",
    "fontweight": "bold",
    "bbox": graybox,
    "fontsize": 10,
}

greenbox = {
    "facecolor": "green",
    "edgecolor": "darkgreen",
    "boxstyle": "Round",
    "linewidth": 2
}
safebutton = {
    "s": "Not Detected",
    "color": "white",
    "fontweight": "bold",
    "bbox": greenbox,
    "fontsize": 10,
}

redbox = {
    "facecolor": "red",
    "edgecolor": "darkred",
    "boxstyle": "Round",
    "linewidth": 2
}
alertbutton = {
    "s": "Detected",
    "color": "white",
    "fontweight": "bold",
    "bbox": redbox,
    "fontsize": 10,
}

buttondic = {
    1: alertbutton,
    0: safebutton,
    -1: graybutton
}


def plot_report(img, len_sus, i, PMCID, SAVE_DIR):
    H, W = img.gray.shape

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle('{}: {}/{}'.format(PMCID, i + 1, len_sus),
                 fontweight='bold', fontsize=12, y=1.02)

    # Input immage #
    ax = fig.add_subplot(521)
    ax.set_title('Input image')
    ax.imshow(cv.cvtColor(img.mat, cv.COLOR_BGR2RGB), aspect="equal")
    ax.axis('off')
    ax = fig.add_subplot(522)
    ax.set_title('Pixel value histogram')
    ax.hist(img.gray.flatten(), bins=256, range=(0, 256))

    # Noise #
    ax = fig.add_subplot(523)
    ax.set_title('Noise')
    ax.imshow(img.no_img)
    ax.axis('off')
    ax = fig.add_subplot(524)
    size = img.size / 1000
    ax.text(0, 0.9, 'File name: {}'.format(img.name))
    ax.text(0, 0.7, 'Image size: {} x {} pixels'.format(H, W))
    ax.text(0, 0.5, 'File size: {:.3g} KB'.format(int(size)))
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
            'Following detection less correct.',
            color='red',
            fontsize=10,
        )
    ax.axis('off')

    # Clipping #
    ax = fig.add_subplot(525)
    ax.set_title('Clipping')
    ax.imshow(cv.cvtColor(img.cl_img, cv.COLOR_BGR2RGB))
    ax.axis('off')
    ax = fig.add_subplot(526)
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
    ax = fig.add_subplot(527)
    ax.set_title('Copy-move')
    ax.imshow(cv.cvtColor(img.cm_img, cv.COLOR_BGR2RGB))
    ax.axis('off')
    ax = fig.add_subplot(528)
    button = buttondic[img.copymove]
    ax.text(0, 0.8, **button)
    if img.copymove is -1:
        ax.text(0, 0.6, 'Not enough keypoints were extracted.', color='red')
    else:
        ax.text(0, 0.6, 'Inlier keypoint ratio: {} %'.format(
            int(img.mask_ratio * 100)))
    ax.axis('off')

    # Cut-Paste #
    ax = fig.add_subplot(529)
    ax.set_title('Cut-paste')
    ax.imshow(img.keyimg[img.gap:-img.gap, img.gap:-img.gap])
    ax.axis('off')
    ax = fig.add_subplot(5, 2, 10)
    button = buttondic[img.cutpaste]
    ax.text(0, 0.8, **button)
    if img.cutpaste is -1:
        ax.text(
            0,
            0.3,
            'Too small image size.\nThe image size must be larger\nthan 488 x 488 pixels.',
            color='red')
    else:
        ax.text(0, 0.6, 'Probability: {} %'.format(int(img.prob * 100)))
    ax.axis('off')

    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(
        os.path.join(
            SAVE_DIR,
            '{}_{}.png'.format(
                PMCID,
                i + 1)),
        bbox_inches='tight',
        dpi=fig.dpi)
    plt.close(fig)


def plot_row(fig, result_imgnames, result_imgarrs, result_ratios, i, j):
    ax = fig.add_subplot(4, 2, 2 * i + 1)
    ax.imshow(cv.cvtColor(result_imgarrs[j], cv.COLOR_BGR2RGB))
    ax.axis('off')
    ax = fig.add_subplot(4, 2, 2 * i + 2)
    ax.text(0, 0.8, **buttondic[1])
    ax.text(0, 0.5, 'Image 1: {}\nImage 2: {}'.format(
        result_imgnames[j][0], result_imgnames[j][1]))
    ax.text(0, 0.3, 'Inlier keypoint ratio: {} %'.format(
        int(result_ratios[j] * 100)))
    ax.axis('off')


def plot_duplication_report(
        result_imgnames,
        result_imgarrs,
        result_ratios,
        PMCID,
        SAVE_DIR):
    len_img = len(result_ratios)
    page = (len_img - 1) // 4
    for p in range(page + 1):
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle('{}: Duplications {}/{}'.format(PMCID, p + 1, page + 1),
                     fontweight='bold', fontsize=12, y=1.02)

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
        fig.savefig(
            os.path.join(
                SAVE_DIR,
                '{}_duplication_{}.png'.format(
                    PMCID,
                    p + 1)),
            bbox_inches='tight',
            dpi=fig.dpi)
        plt.close(fig)


def detection(
        MODEL_DIR,
        REPORT_DIR,
        PMC_DIRs,
        log_file_name,
        wave_thres=488,
        gap=32):
    # Detectors #
    detector_cl = Clipping()
    detector_no = Noise(model_name=MODEL_DIR + 'noise_oneclass_42.sav')
    detector_cm = CopyMove(min_kp=20, min_match=20, min_key_ratio=0.75)
    detector_du = Duplication(min_kp=20, min_match=20, min_key_ratio=0.75)
    detector_cp = CutPaste(
        model_name=MODEL_DIR +
        'cutpaste_svm_uci_200.sav',
        param_name=MODEL_DIR +
        'cutpaste_svm_uci_200.sav-param.npz',
    )

    log = []
    for PMC_DIR in PMC_DIRs:
        OUT_DIR = os.path.join(PMC_DIR, 'output')
        if os.path.exists(OUT_DIR):
            shutil.rmtree(OUT_DIR)
        os.makedirs(OUT_DIR)

        PMCID = os.path.basename(PMC_DIR)

        # Load images #
        images_path = glob.glob(
            os.path.join(
                PMC_DIR,
                'photo',
                '*.png'))
        if len(images_path) is 0:
            print(PMC_DIR, "No images.")
            log.append([PMCID, 0, 0, 0, 0, 0, 0, 0])
            continue
        suspicious_images = [
            SuspiciousImage(
                path,
                hist_eq=True,
                algorithm='orb',
                nfeatures=2000,
                gap=gap) for path in images_path]
        len_sus = len(suspicious_images)

        # Report #
        report = pd.DataFrame(
            0,
            index=[
                img.name for img in suspicious_images],
            columns=[
                'Clipping',
                'area_ratio',
                'CopyMove',
                'mask_ratio',
                'CutPaste',
                'proba',
                'Duplication', ])

        for i in range(len_sus):
            img = suspicious_images[i]
            imgname = img.name

            # Paint-out (blown-out highlights, clipping) check #
            pred = detector_cl.detect(img)
            report.loc[imgname, 'Clipping'] = pred
            img.clipping = pred
            if pred is 1:
                ratio = detector_cl.ratio_
                report.loc[imgname, 'area_ratio'] = ratio
                img.area_ratio = ratio
                file_name = os.path.join(
                    OUT_DIR, '{}_clipping.jpg'.format(imgname))
                detector_cl.save_image(file_name)
                img.cl_img = detector_cl.image_

            # Copy-move check #
            pred = detector_cm.detect(img)
            report.loc[imgname, 'CopyMove'] = pred
            img.copymove = pred
            if pred is 1:
                ratio = detector_cm.mask_ratio_
                report.loc[imgname, 'mask_ratio'] = ratio
                img.mask_ratio = ratio
                file_name = os.path.join(
                    OUT_DIR, '{}_copymove.jpg'.format(imgname))
                detector_cm.save_image(file_name)
                img.cm_img = detector_cm.image_

        # Noise check #
        report['Noise'] = detector_no.detect(suspicious_images)
        report['dist'] = detector_no.dist_
        for i, img in enumerate(suspicious_images):
            img.noise = report.Noise[i]

        # Cut-paste check #
        enough_size_images = [
            img for img in suspicious_images
            if img.gray.shape[0] >= wave_thres and img.gray.shape[1] >= wave_thres]
        if len(enough_size_images) > 0:
            results = detector_cp.detect(enough_size_images)
            probas = detector_cp.proba_
            for i, img in enumerate(enough_size_images):
                img.cutpaste = results[i]
                img.prob = probas[i]
        for i, img in enumerate(suspicious_images):
            report.iloc[i, 4] = img.cutpaste
            report.iloc[i, 5] = img.prob

        # Duplication check #
        # flip_images = [
        #     SuspiciousImage().make_flip(img.mat, img.name + '-flip')
        #     for img in suspicious_images]
        result_imgnames = []
        result_imgarrs = []
        result_ratios = []
        for i in range(len_sus):
            img = suspicious_images[i]
            imgname = img.name
            for j in range(i + 1, len_sus):
                pred = detector_du.detect(
                    img, suspicious_images[j])
                report.loc[imgname, 'Duplication'] += pred
                if pred is 1:
                    file_name = os.path.join(
                        OUT_DIR, '{}_{}_duplication.jpg'.format(
                            imgname, suspicious_images[j].name))
                    detector_du.save_image(file_name)
                    result_imgnames.append(
                        [imgname, suspicious_images[j].name])
                    result_imgarrs.append(detector_du.image_)
                    result_ratios.append(detector_du.mask_ratio_)

                # # flipped images
                # pred = detector_du.detect(
                #     img, flip_images[j])
                # report.loc[imgname, 'Duplication'] += pred
                # if pred:
                #     file_name = os.path.join(
                #         OUT_DIR, '{}_{}_duplication.jpg'.format(
                #             imgname, flip_images[j].name))
                #     detector_du.save_image(file_name)
                #     result_imgnames.append([imgname, flip_images[j].name])
                #     result_imgarrs.append(detector_du.image_)
                #     result_ratios.append(detector_du.mask_ratio_)

        # Output report #
        print(PMC_DIR, len(glob.glob(os.path.join(OUT_DIR, '*.jpg'))))
        report.to_csv(
            os.path.join(
                PMC_DIR,
                'report_{}.csv'.format(
                    os.path.basename(PMC_DIR))))

        report = report[report == 1].sum()
        len_output = len(glob.glob(os.path.join(OUT_DIR, '*.jpg')))
        log.append([PMCID,
                    len_sus,
                    len_output,
                    report['Clipping'],
                    report['CopyMove'],
                    report['Duplication'],
                    report['CutPaste'],
                    report['Noise']])

        for i, img in enumerate(suspicious_images):
            plot_report(img, len_sus, i, PMCID, REPORT_DIR)

        if len(result_ratios) > 0:
            plot_duplication_report(
                result_imgnames,
                result_imgarrs,
                result_ratios,
                PMCID,
                REPORT_DIR)

    log = pd.DataFrame(log)
    log.columns = [
        'PMCID',
        'Extracted',
        'Suspicious',
        'Clipping',
        'CopyMove',
        'Duplication',
        'CutPaste',
        'Noise']
    log.to_csv(os.path.join(DIR, log_file_name))


if __name__ == "__main__":
    MODEL_DIR = '/Users/yuki/OneDrive - The University of Tokyo/imdetector/model/'
    DIR = '/Volumes/data-yuki/retraction_watch/190722/'
    start = time.time()

    # REPORT_DIR = "/Volumes/data-yuki/retraction_watch/190722/retracted_report/"
    # PMC_DIRs = sorted(glob.glob(os.path.join(DIR, 'retracted', 'PMC*')))
    # detection(MODEL_DIR, REPORT_DIR, PMC_DIRs, 'retracted259.csv')

    REPORT_DIR = DIR + 'negative_report'
    PMC_DIRs = sorted(
        glob.glob(
            os.path.join(
                DIR,
                'negative',
                'PMC*')))
    # # PMC_DIRs = [DIR + 'negative/PMC419710',
    # #             DIR + 'negative/PMC1828703',
    # #             DIR + 'negative/PMC2447896',
    # #             DIR + 'negative/PMC3538858',
    # #             DIR + 'negative/PMC3885421',
    # #             DIR + 'negative/PMC3900932',
    # #             DIR + 'negative/PMC4052923',
    # #             DIR + 'negative/PMC4689595',
    # #             DIR + 'negative/PMC5209699',
    # #             DIR + 'negative/PMC5558095',
    # #             ]
    detection(MODEL_DIR, REPORT_DIR, PMC_DIRs, 'negative259.csv')
    print('TIME: {} h'.format((time.time() - start) / 360))
