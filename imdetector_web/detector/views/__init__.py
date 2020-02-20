import os
import shutil

# from django.views.generic import FormView

from django.shortcuts import render, redirect, get_object_or_404

from detector.forms import FileForm
from detector.models import File, Suspicious, SuspiciousDuplication
from .detection import detection

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
print('BASE_DIR: ', BASE_DIR)
# MODEL_DIR = os.path.join(BASE_DIR, 'model')
EXTRACT_DIR = os.path.join(BASE_DIR, 'media', 'extracts')
RESULT_DIR = os.path.join(BASE_DIR, 'media', 'results')


def index(req):
    if req.method == 'GET':
        return render(req, 'index.html', {
            'form': FileForm(),
            'files': File.objects.all(),
            'results': Suspicious.objects.all()
        })


def about(req):
    return render(req, 'detector/about.html')


def contact(req):
    return render(req, 'detector/contact.html')


def download(req):
    return render(req, 'detector/download.html')


def progress(req):
    form = FileForm(req.POST, req.FILES)
    if not form.is_valid():
        raise ValueError('invalid form')

    """ 前のデータを削除 """
    try:
        File.objects.all().delete()
        Suspicious.objects.all().delete()
        SuspiciousDuplication.objects.all().delete()
    except BaseException:
        print('no data')

    """ ファイル登録 """
    file = File()
    file.zip = form.cleaned_data['file']
    file.save()
    print('uploaded')

    return redirect('post', post_id=file.pk)


def post(req, post_id):
    try:
        Suspicious.objects.all().delete()
        SuspiciousDuplication.objects.all().delete()
    except BaseException:
        print('no data')

    len_sus, n_dp = detection(post_id)

    return render(req,
                  'detector/result.html',
                  {'post_id': post_id,
                   'results': Suspicious.objects.all(),
                   'duplication': SuspiciousDuplication.objects.all(),
                   'n': len_sus,
                   'n_dp': n_dp})
