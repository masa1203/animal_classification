from django.db import models

# Create your models here.
class AnimalImage(models.Model):

    class Meta:
        db_table = 'animal_image'
    
    animal_image = models.ImageField(
        verbose_name="イメージ",
        blank=True, null=True,
        upload_to='images/'
    )