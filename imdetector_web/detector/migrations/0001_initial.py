# Generated by Django 3.0.2 on 2020-02-05 08:18

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='File',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('zip', models.FileField(upload_to='files/')),
            ],
        ),
        migrations.CreateModel(
            name='Suspicious',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('post_id', models.IntegerField(default=0, null=True)),
                ('name', models.CharField(max_length=256, null=True)),
                ('size', models.IntegerField(default=0, null=True)),
                ('h', models.IntegerField(default=0, null=True)),
                ('w', models.IntegerField(default=0, null=True)),
                ('original', models.ImageField(upload_to='results/<django.db.models.fields.IntegerField>/')),
                ('noise', models.IntegerField(default=0, null=True)),
                ('no_img', models.ImageField(upload_to='results/<django.db.models.fields.IntegerField>/')),
                ('clipping', models.IntegerField(default=0, null=True)),
                ('cl_img', models.ImageField(upload_to='results/<django.db.models.fields.IntegerField>/')),
                ('area_ratio', models.IntegerField(default=0, null=True)),
                ('copymove', models.IntegerField(default=-1, null=True)),
                ('cm_img', models.ImageField(upload_to='results/<django.db.models.fields.IntegerField>/')),
                ('mask_ratio', models.IntegerField(default=0, null=True)),
                ('cutpaste', models.IntegerField(default=-1, null=True)),
                ('cp_img', models.ImageField(upload_to='results/<django.db.models.fields.IntegerField>/')),
                ('prob', models.IntegerField(default=0, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='SuspiciousDuplication',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('post_id', models.IntegerField(default=0, null=True)),
                ('name1', models.CharField(max_length=256, null=True)),
                ('name2', models.CharField(max_length=256, null=True)),
                ('du_img', models.ImageField(upload_to='results/<django.db.models.fields.IntegerField>/')),
                ('mask_ratio', models.IntegerField(default=0, null=True)),
            ],
        ),
    ]
