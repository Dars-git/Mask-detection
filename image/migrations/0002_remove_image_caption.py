# Generated by Django 4.0.3 on 2022-03-11 12:16

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('image', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='image',
            name='caption',
        ),
    ]
