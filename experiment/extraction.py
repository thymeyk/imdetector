import glob
import os
import shutil

from imdetector.image import SuspiciousImage
from imdetector.photopick import PhotoPick
from imdetector.utils import mkdir
from imdetector.dismantler import Dismantler


def extraction(PMC_DIRs):
    detector = PhotoPick()

    for PMC_DIR in PMC_DIRs:
        print(PMC_DIR)
        # from JPEG images #
        image_files = glob.glob(os.path.join(PMC_DIR, '*.jpg'))
        if len(image_files) > 0:
            SAVE_DIR = os.path.join(PMC_DIR, 'from_jpg')
            if os.path.isdir(SAVE_DIR):
                shutil.rmtree(SAVE_DIR)
            SAVE_DIR = mkdir(SAVE_DIR)
            Dismantler().dismantle(image_files, SAVE_DIR)

        # # from PDF files
        # pdf_file = glob.glob(os.path.join(pmcDIR, '*.pdf'))
        # if len(pdf_file) > 0:
        #     pdf_file = pdf_file[0]
        #     saveDIR = mkdir(os.path.join(pmcDIR, 'from_pdf'))
        #     extract_img_from_pdf(pdf_file, saveDIR)

        subimgs_path = glob.glob(
            os.path.join(
                PMC_DIR,
                'from_jpg',
                'subimg_cut',
                '*.png'))
        photo_DIR = os.path.join(PMC_DIR, 'photo')
        if os.path.isdir(photo_DIR):
            shutil.rmtree(photo_DIR)
        photo_DIR = mkdir(photo_DIR)
        other_DIR = mkdir(os.path.join(PMC_DIR, 'from_jpg', 'other'))
        suspicious_imgs = [
            SuspiciousImage(img_path) for img_path in subimgs_path]
        if len(suspicious_imgs) is 0:
            print("No images")
            continue
        pred = detector.detect(suspicious_imgs)
        for i, p in enumerate(pred):
            if p == 1:
                shutil.copy(subimgs_path[i], photo_DIR)
            elif p == 0:
                shutil.copy(subimgs_path[i], other_DIR)


if __name__ == "__main__":
    DIR = '/Volumes/data-yuki/retraction_watch/190722/'
    # PMC_DIRs = glob.glob(os.path.join(DIR, 'negative_controll', 'PMC*'))
    # extraction(PMC_DIRs)
