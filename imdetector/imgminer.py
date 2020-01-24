import glob
import os
import cv2 as cv
import numpy as np
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTContainer, LTImage
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from imdetector.dismantler import Dismantler


def find_images_recursively(layout_obj):
    """
    Find LTImage recursively.

    :param LTContainer layout_obj:
    :rtype: list
    :return: list of LTImage
    """

    if isinstance(layout_obj, LTImage):
        return [layout_obj]
    if isinstance(layout_obj, LTContainer):
        images = []
        for child in layout_obj:
            images.extend(find_images_recursively(child))
        return images
    return []


def ltimage2imgarray(ltimage):
    """
    Convert LTImage to image array.

    :param LTImage ltimage:
    :rtype: np.ndarray
    :return: imgarr
    """

    imgarr = None
    if ltimage.stream:
        file_stream = ltimage.stream.get_rawdata()
        array = np.asarray(bytearray(file_stream), dtype=np.uint8)
        imgarr = cv.imdecode(array, -1)

    return imgarr


def imgminer(pdf_path, OUT_DIR, save=True, file_ext='png'):
    """
    Extract images from pdf file using PDFMiner (https://euske.github.io/pdfminer/programming.html)

    :param str pdf_path:
    :param str OUT_DIR:
    :param bool save:
    :param str file_ext:
    :rtype: list
    :return: list of images
    """

    resource_manager = PDFResourceManager()
    device = PDFPageAggregator(resource_manager)
    interpreter = PDFPageInterpreter(resource_manager, device)

    images = []
    with open(pdf_path, 'rb') as f:
        # 1ページずつ処理
        for p, page in enumerate(PDFPage.get_pages(f)):
            interpreter.process_page(page)
            layout = device.get_result()  # LTPage object

            ltimages = find_images_recursively(layout)  # list of LTImage

            # Sort based on coordinates
            ltimages.sort(key=lambda b: (-b.y1, b.x0))

            for ltimage in ltimages:
                imgarr = ltimage2imgarray(ltimage)
                if imgarr is not None:
                    images.append(imgarr)
                    if save:
                        file_name = 'page{}_{}.{}'.format(
                            p, ltimage.name, file_ext)
                        cv.imwrite(os.path.join(OUT_DIR, file_name), imgarr)
    return images


def extract_img_from_pdf(pdf_file, SAVE_DIR):
    IMG_DIR = os.path.join(SAVE_DIR, 'img')
    os.makedirs(IMG_DIR, exist_ok=True)
    images = imgminer(pdf_file, OUT_DIR=IMG_DIR)
    files = glob.glob(os.path.join(IMG_DIR, '*.png'))
    print(os.path.join(IMG_DIR, '*.png'))
    print(files)
    thresholds = {
        'blackThres': 0.8,
        'whiteThres': 0.8,
        'white2Thres': 0.001,
        'whiteblackThres': 0.9,
        'areaThres': 1000,
        'splitThres': 0.999,
        'varThres': 0,
        'var2Thres': 100}
    Dismantler(thresholds=thresholds).dismantle(files, SAVE_DIR)
