# Generated by Django 4.0 on 2023-11-13 07:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0008_alter_predresults_cet'),
    ]

    operations = [
        migrations.AlterField(
            model_name='predresults',
            name='cet',
            field=models.FloatField(default=90),
            preserve_default=False,
        ),
    ]
