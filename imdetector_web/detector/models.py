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
        result = Photo.objects.filter(pk=self.pk)
        previous = result[0] if len(result) else None
        super(Photo, self).save()

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


class Photo(models.Model):
    @delete_previous_file
    def save(
            self,
            force_insert=False,
            force_update=False,
            using=None,
            update_fields=None):
        super(Photo, self).save()

    @delete_previous_file
    def delete(self, using=None, keep_parents=False):
        super(Photo, self).delete()

    name = models.CharField(max_length=256, null=True)
    title = models.CharField(max_length=256, null=True)
    paintout = models.ImageField(upload_to='results/')
    area_ratio = models.IntegerField(null=True, default=0)
    copymove = models.ImageField(upload_to='results/')
    mask_ratio = models.IntegerField(null=True, default=0)
