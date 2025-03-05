from django.db import models

class AcceptedTitle(models.Model):
    title_name = models.TextField(max_length=255,db_index=True,unique=True)
    owner_name = models.TextField(max_length=100,db_index=True)
    state = models.TextField(max_length=20,db_index=True)
    publication_city_district = models.TextField()


    class Meta:
        db_table = "accepted_titles"  # Explicitly set the table name
        indexes=[models.Index(fields=['title_name'])]

    def __str__(self):
        return f"{self.title_name} ({self.title_code})"
