from django.db import models

# Create your models here.
image=''
class Image(models.Model):
    image=models.ImageField()
    def __str__(self):
        return self.image

