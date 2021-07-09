from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Consumer(models.Model):
	user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
	points = models.IntegerField(default=0)
	about = models.CharField(max_length=200, default="I am a Environment Lover")

	def __str__(self):
		return self.user.__str__()