from django.db import models

# Create your models here.
class PredictionModel(models.Model):
	image = models.ImageField(upload_to='images/')