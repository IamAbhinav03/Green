# Generated by Django 3.2.4 on 2021-06-18 20:31

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='predictionmodel',
            old_name='img',
            new_name='image',
        ),
    ]
