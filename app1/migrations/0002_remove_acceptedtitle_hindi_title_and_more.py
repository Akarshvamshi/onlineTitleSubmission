# Generated by Django 5.1.6 on 2025-03-04 17:01

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("app1", "0001_initial"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="acceptedtitle",
            name="hindi_title",
        ),
        migrations.RemoveField(
            model_name="acceptedtitle",
            name="register_serial_no",
        ),
        migrations.RemoveField(
            model_name="acceptedtitle",
            name="regn_no",
        ),
        migrations.RemoveField(
            model_name="acceptedtitle",
            name="title_code",
        ),
    ]
