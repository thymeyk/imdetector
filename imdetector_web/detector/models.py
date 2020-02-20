from django.db import models
import os


def delete_previous_file(function):
    """
    不要となる古いファイルを削除する為のデコレータ実装.

    :param function: メイン関数
    :return: wrapper
    """
    def wrapper(*args, **kwargs):
        """
        Wrapper 関数.

        :param args: 任意の引数
        :param kwargs: 任意のキーワード引数
        :return: メイン関数実行結果
        """
        self = args[0]

        # 保存前のファイル名を取得
        result = Suspicious.objects.filter(pk=self.pk)
        previous = result[0] if len(result) else None
        super(Suspicious, self).save()

        # 関数実行
        result = function(*args, **kwargs)

        # 保存前のファイルがあったら削除
        if previous:
            MEDIA_ROOT = os.path.join(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__))),
                'media')
            os.remove(MEDIA_ROOT + '/' + previous.image.name)
        return result
    return wrapper


class File(models.Model):
    zip = models.FileField(upload_to='files/')


class Suspicious(models.Model):
    @delete_previous_file
    def save(
            self,
            force_insert=False,
            force_update=False,
            using=None,
            update_fields=None):
        super(Suspicious, self).save()

    @delete_previous_file
    def delete(self, using=None, keep_parents=False):
        super(Suspicious, self).delete()

    post_id = models.IntegerField(null=True, default=0)

    name = models.CharField(max_length=256, null=True)
    size = models.IntegerField(null=True, default=0)
    h = models.IntegerField(null=True, default=0)
    w = models.IntegerField(null=True, default=0)
    original = models.ImageField(upload_to='results/{}/'.format(post_id))

    noise = models.IntegerField(null=True, default=0)
    no_img = models.ImageField(upload_to='results/{}/'.format(post_id))

    clipping = models.IntegerField(null=True, default=0)
    cl_img = models.ImageField(upload_to='results/{}/'.format(post_id))
    area_ratio = models.IntegerField(null=True, default=0)

    copymove = models.IntegerField(null=True, default=-1)
    cm_img = models.ImageField(upload_to='results/{}/'.format(post_id))
    mask_ratio = models.IntegerField(null=True, default=0)

    cutpaste = models.IntegerField(null=True, default=-1)
    cp_img = models.ImageField(upload_to='results/{}/'.format(post_id))
    prob = models.IntegerField(null=True, default=0)


class SuspiciousDuplication(models.Model):
    post_id = models.IntegerField(null=True, default=0)
    name1 = models.CharField(max_length=256, null=True)
    name2 = models.CharField(max_length=256, null=True)
    du_img = models.ImageField(upload_to='results/{}/'.format(post_id))
    mask_ratio = models.IntegerField(null=True, default=0)
