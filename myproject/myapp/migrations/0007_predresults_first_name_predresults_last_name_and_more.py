# Generated by Django 4.0 on 2023-11-13 07:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0006_alter_predresults_recommended_course'),
    ]

    operations = [
        migrations.AddField(
            model_name='predresults',
            name='first_name',
            field=models.CharField(max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='predresults',
            name='last_name',
            field=models.CharField(max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='predresults',
            name='sex',
            field=models.CharField(max_length=100, null=True),
        ),
    ]
