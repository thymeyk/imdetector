import glob
import os
import shutil

from imdetector.image import SuspiciousImage
from imdetector.photopick import PhotoPick
from imdetector.utils import mkdir
from imdetector.dismantler import Dismantler
from imdetector.imgminer import extract_img_from_pdf


def extraction(PMC_DIRs, from_where='from_pdf'):
    detector = PhotoPick()

    for PMC_DIR in PMC_DIRs:
        print(PMC_DIR)
        # from PDF files
        files = glob.glob(os.path.join(PMC_DIR, '*.pdf'))
        if len(files) > 0:
            pdf_file = files[0]
            SAVE_DIR = mkdir(os.path.join(PMC_DIR, from_where))
            extract_img_from_pdf(pdf_file, SAVE_DIR)

        # # from JPEG images #
        # image_files = glob.glob(os.path.join(PMC_DIR, '*.jpg'))
        # if len(image_files) > 0:
        #     SAVE_DIR = mkdir(PMC_DIR, 'from_jpg')
        #     Dismantler().dismantle(image_files, SAVE_DIR)

        subimgs_path = glob.glob(
            os.path.join(
                PMC_DIR,
                from_where,
                'subimg_cut',
                '*.png'))
        photo_DIR = mkdir(os.path.join(PMC_DIR, 'photo'))
        other_DIR = mkdir(os.path.join(PMC_DIR, from_where, 'other'))
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
    PMC_DIRs = glob.glob(os.path.join(DIR, 'negative2', 'PMC*'))
    extraction(PMC_DIRs, 'from_pdf')
