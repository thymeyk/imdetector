import glob
import os
import zipfile

from django.shortcuts import render
from .forms import FileForm
from .models import File, Photo


BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
print('BASE_DIR: ', BASE_DIR)
OUT_DIR = os.path.join(BASE_DIR, 'media', 'images')
RESULT_DIR = os.path.join(BASE_DIR, 'media', 'results')


def index(req):
    if req.method == 'GET':
        return render(req, 'index.html', {
            'form': FileForm(),
            'files': File.objects.all(),
            'results': Photo.objects.all()
        })


def about(req):
    return render(req, 'detector/about.html')


def contact(req):
    return render(req, 'detector/contact.html')


def download(req):
    return render(req, 'detector/download.html')


def result(req):
    form = FileForm(req.POST, req.FILES)
    if not form.is_valid():
        raise ValueError('invalid form')

    # # 前のデータを削除
    # try:
    #     File.objects.all().delete()
    #     Photo.objects.all().delete()
    # except BaseException:
    #     print('no data')
    # if os.path.exists(OUT_DIR):
    #     shutil.rmtree(OUT_DIR)
    # if os.path.exists(RESULT_DIR):
    #     shutil.rmtree(RESULT_DIR)
    #     os.mkdir(RESULT_DIR)

    # ファイル登録
    file = File()
    file.zip = form.cleaned_data['file']
    file.save()

    # zipファイル展開
    if file.zip.path.split('.')[-1] == 'zip':
        with zipfile.ZipFile(os.path.join(BASE_DIR, file.zip.path)) as existing_zip:
            existing_zip.extractall(OUT_DIR)

        images_path = glob.glob(os.path.join(
            glob.glob(os.path.join(OUT_DIR, '*'))[0], '*'))
        images_url = list(map(lambda x: x.split('media/')[-1], images_path))
        # images_name = list(map(lambda x: x.split(
        #     '/')[-1].split('.')[0], images_url))
        print(images_url)
    return render(req, 'detector/result.html',
                  {'results': Photo.objects.all()})

# #
# #         images = [Image(path, isHist=True, algorithm='orb', nfeatures=5000) for path in images_path]
# #     print('loaded')
# #
# #     # pdfファイル展開
# #     if file.zip.path.split('.')[-1] == 'pdf' or file.zip.path.split('.')[-1] == 'PDF':
# #         images = imgminer(file.zip.path, OUT_DIR)
# #
# #     print(images)
# #
# #     # """ 画像の切り出し """
# #     # images = splitting(images)
# #
# #     # """ 結果の画像の登録 """
# #     # ### Duplication check ###
# #     # for i in range(len(images)):
# #     #     for j in range(i + 1, len(images)):
# #     #         isDetected, result_path = Duplication().detector(
# #     #             images[j], images[i])
# #     #         if isDetected:
# #     #             result_url = result_path.split('media/')[-1]
# #     #             photo = Photo()
# #     #             photo.name = images[i].name.split('/')[-1] + ' & ' + images[j].name.split('/')[-1]
# #     #             photo.result = result_url
# #     #             photo.title = 'Reuse'
# #     #             photo.ratio = 50
# #     #             photo.save()
# #     # print('DONE: Duplication check')
# #     #
# #     # for i in range(len(images)):
# #     #     imgname = images[i].name.split('/')[-1]
# #     #     img = images[i]
# #     #
# #     #
# #     #     ### Painting-out check ###
# #     #     isPaintingOut, img_detect = paintout(images[i])
# #     #     if isPaintingOut:
# #     #         result_path = os.path.join(RESULT_DIR, imgname.split('.')[0] + '_po.jpg')
# #     #         result_url = result_path.split('media/')[-1]
# #     #         cv.imwrite(result_path, img_detect)
# #     #         photo = Photo()
# #     #         photo.name = imgname
# #     #         photo.result = result_url
# #     #         photo.title = 'Over-adjustment of contrast/brightness or painting-out'
# #     #         photo.ratio = 50
# #     #         photo.save()
# #     #         continue
# #     #     print('paintingout')
# #     #
# #     #     ### Copy-move check ###
# #     #     isCopyMove, img_detect = copymove(img)
# #     #     if isCopyMove:
# #     #         result_path = os.path.join(Duplication.RESULT_DIR, imgname.split('.')[0] + '_cm.jpg')
# #     #         result_url = result_path.split('media/')[-1]
# #     #         cv.imwrite(result_path, img_detect)
# #     #         photo = Photo()
# #     #         photo.name = imgname
# #     #         photo.result = result_url
# #     #         photo.title = 'Reuse within a same image'
# #     #         photo.ratio = 50
# #     #         photo.save()
# #     #     print('copymove')
# #     #
# #     #     print('DONE: ', imgname)
# #     #
# #     # print(Photo.objects.all())
# #
