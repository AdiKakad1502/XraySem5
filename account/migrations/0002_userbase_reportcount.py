# Generated by Django 4.1.2 on 2022-11-27 17:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('account', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='userbase',
            name='ReportCount',
            field=models.IntegerField(blank=True, default=0),
        ),
    ]
