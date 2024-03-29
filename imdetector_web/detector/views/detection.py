import os
import cv2 as cv
import glob
import shutil
import zipfile

from imdetector.dismantler import Dismantler
from imdetector.image import SuspiciousImage
from imdetector.photopick import PhotoPick
from imdetector.noise import Noise
from imdetector.clipping import Clipping
from imdetector.copymove import CopyMove, Duplication
from imdetector.cutpaste import CutPaste

from detector.models import File, Suspicious, SuspiciousDuplication

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


def detection(post_id):
    """ zipファイル展開 """
    file = File.objects.get(pk=post_id)
    p_id = post_id % 10
    OUT_DIR = os.path.join(
        BASE_DIR,
        'media',
        'images',
        str(p_id))
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    if file.zip.path.split('.')[-1] == 'zip':
        with zipfile.ZipFile(os.path.join(BASE_DIR, file.zip.path)) as existing_zip:
            existing_zip.extractall(OUT_DIR)
        # TODO: fix
        images_path = glob.glob(os.path.join(OUT_DIR, '*', '*'))
        images_path = [p for p in images_path if p.split(
            '.')[-1] in ['jpg', 'png', 'tif', 'JPG', 'JPEG', 'TIF']]
        images_url = list(map(lambda x: x.split('media/')[-1], images_path))
        # images_name = list(map(lambda x: x.split(
        #     '/')[-1].split('.')[0], images_url))
        print(images_url)
    elif file.zip.path.split('.')[-1] in ['jpg', 'png', 'tif', 'JPG', 'JPEG', 'TIF']:
        images_path = [file.zip.path]
    else:
        return 0
    print('loaded')

    RESULT_DIR = os.path.join(
        BASE_DIR,
        'media',
        'results',
        str(p_id)
    )
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR)

    # # pdfファイル展開
    # if file.zip.path.split('.')[-1] == 'pdf' or file.zip.path.split('.')[-1] == 'PDF':
    #     images = imgminer(file.zip.path, OUT_DIR)

    # """ 図の分割 """
    # if len(images_path) > 0:
    #     Dismantler().dismantle(images_path, EXTRACT_DIR)
    # subimgs_path = glob.glob(os.path.join(EXTRACT_DIR, 'subimg_cut', '*.png'))
    # suspicious_images = [
    #     SuspiciousImage(img_path) for img_path in subimgs_path]
    # if len(suspicious_images) is 0:
    #     print("No images")
    #     return 0
    # detector = PhotoPick(
    #     model_name=os.path.join(
    #         MODEL_DIR, 'photopicker_rf_lee_2700.sav'),
    #     param_name=os.path.join(
    #         MODEL_DIR, 'photopicker_rf_lee_2700.sav-param.npz'))
    # pred = detector.detect(suspicious_images)
    # suspicious_images = [img for i, img in enumerate(
    #     suspicious_images) if pred[i] == 1]

    suspicious_images = [
        SuspiciousImage(
            path,
            nfeatures=5000) for path in images_path]

    print(suspicious_images)
    len_sus = len(suspicious_images)

    # """ 画像の切り出し """
    # images = splitting(images)

    """ Detection """
    # Detectors #
    # detector_no = Noise(
    #     model_name=os.path.join(
    #         MODEL_DIR,
    #         'noise_oneclass_42.sav'))
    detector_cl = Clipping(min_area=100)
    detector_cm = CopyMove(
        min_kp=20,
        min_match=20,
        min_key_ratio=0.75,
        flags=0)
    detector_du = Duplication(
        min_kp=20,
        min_match=20,
        min_key_ratio=0.75,
        flags=0)
    # detector_cp = CutPaste(
    #     model_name=os.path.join(
    #         MODEL_DIR, 'cutpaste_svm_uci_200.sav'), param_name=os.path.join(
    #         MODEL_DIR, 'cutpaste_svm_uci_200.sav-param.npz'), )

    for img in suspicious_images:
        dsize = (img.w, img.h)

        # Clipping check #
        pred = detector_cl.detect(img)
        img.clipping = pred
        if pred is 1:
            ratio = detector_cl.ratio_
            img.area_ratio = ratio
            img.cl_img = cv.resize(detector_cl.image_, dsize=dsize)

        # Copy-move check #
        pred = detector_cm.detect(img)
        img.copymove = pred
        if pred is 1:
            ratio = detector_cm.mask_ratio_
            img.mask_ratio = ratio
            img.cm_img = cv.resize(detector_cm.image_, dsize=dsize)
    print('detected')

    for img in suspicious_images:
        nameroot = img.name
        dsize = (img.w, img.h)

        suspicious = Suspicious()
        suspicious.post_id = p_id
        suspicious.w, suspicious.h = dsize
        suspicious.name = img.name
        suspicious.size = img.size
        file_name = os.path.join(
            RESULT_DIR, '{}.jpg'.format(nameroot))
        cv.imwrite(file_name, cv.resize(img.mat, dsize=dsize))
        suspicious.original = file_name.split('media/')[-1]

        # suspicious.noise = img.noise
        file_name = os.path.join(
            RESULT_DIR, '{}_no.jpg'.format(nameroot))
        cv.imwrite(file_name, img.no_img)
        suspicious.no_img = file_name.split('media/')[-1]
        # suspicious.no_img = img.no_img

        suspicious.clipping = img.clipping
        if img.clipping:
            file_name = os.path.join(
                RESULT_DIR, '{}_cl.jpg'.format(nameroot))
            cv.imwrite(file_name, img.cl_img)
            suspicious.cl_img = file_name.split('media/')[-1]
        else:
            suspicious.cl_img = suspicious.original
        suspicious.area_ratio = int(img.area_ratio * 100)

        suspicious.copymove = img.copymove
        if suspicious.copymove:
            file_name = os.path.join(
                RESULT_DIR, '{}_cm.jpg'.format(nameroot))
            cv.imwrite(file_name, img.cm_img)
            suspicious.cm_img = file_name.split('media/')[-1]
        else:
            suspicious.cm_img = suspicious.original
        suspicious.mask_ratio = int(img.mask_ratio * 100)

        # suspicious.cutpaste = img.cutpaste
        file_name = os.path.join(
            RESULT_DIR, '{}_cp.jpg'.format(nameroot))
        cv.imwrite(file_name, cv.resize(
            img.keyimg[img.gap:-img.gap, img.gap:-img.gap], dsize=dsize))
        suspicious.cp_img = file_name.split('media/')[-1]

        suspicious.save()
    print('saved')

    ### Duplication check ###
    n_dp = 0
    for i in range(len_sus):
        img = suspicious_images[i]
        imgname = img.name
        for j in range(i + 1, len_sus):
            pred = detector_du.detect(
                suspicious_images[j], img)
            if pred is 1:
                file_name = os.path.join(
                    RESULT_DIR, '{}_{}_duplication.jpg'.format(
                        suspicious_images[j].name, imgname))
                detector_du.save_image(file_name)
                suspicious = SuspiciousDuplication()
                suspicious.post_id = p_id
                suspicious.name1 = suspicious_images[j].name
                suspicious.name2 = imgname
                suspicious.du_img = file_name.split('media/')[-1]
                suspicious.mask_ratio = int(detector_du.mask_ratio_ * 100)
                suspicious.save()
                n_dp = n_dp + 1

    # for i in range(len(images)):
    #     imgname = images[i].name.split('/')[-1]
    #     img = images[i]
    #
    #
    #     ### Painting-out check ###
    #     isPaintingOut, img_detect = paintout(images[i])
    #     if isPaintingOut:
    #         result_path = os.path.join(RESULT_DIR, imgname.split('.')[0] + '_po.jpg')
    #         result_url = result_path.split('media/')[-1]
    #         cv.imwrite(result_path, img_detect)
    #         photo = Photo()
    #         photo.name = imgname
    #         photo.result = result_url
    #         photo.title = 'Over-adjustment of contrast/brightness or painting-out'
    #         photo.ratio = 50
    #         photo.save()
    #         continue
    #     print('paintingout')
    #
    #     ### Copy-move check ###
    #     isCopyMove, img_detect = copymove(img)
    #     if isCopyMove:
    #         result_path = os.path.join(Duplication.RESULT_DIR, imgname.split('.')[0] + '_cm.jpg')
    #         result_url = result_path.split('media/')[-1]
    #         cv.imwrite(result_path, img_detect)
    #         photo = Photo()
    #         photo.name = imgname
    #         photo.result = result_url
    #         photo.title = 'Reuse within a same image'
    #         photo.ratio = 50
    #         photo.save()
    #     print('copymove')
    #
    #     print('DONE: ', imgname)
    #
    # print(Photo.objects.all())

    return len_sus, n_dp
