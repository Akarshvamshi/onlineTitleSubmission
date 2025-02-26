from django.db import models

class AcceptedTitle(models.Model):
    title_code = models.TextField()
    title_name = models.TextField()
    hindi_title = models.TextField()
    register_serial_no = models.TextField()
    regn_no = models.TextField()
    owner_name = models.TextField()
    state = models.TextField()
    publication_city_district = models.TextField()
    periodity = models.TextField()

    class Meta:
        db_table = "accepted_titles"  # Explicitly set the table name

    def __str__(self):
        return f"{self.title_name} ({self.title_code})"
