# Generated by Django 2.0.1 on 2018-03-13 19:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('recomendador', '0005_auto_20180313_1305'),
    ]

    operations = [
        migrations.AlterField(
            model_name='usuario',
            name='direccionip',
            field=models.CharField(default='MAL!', max_length=50),
        ),
    ]