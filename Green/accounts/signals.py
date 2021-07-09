from django.dispatch import receiver
from django.db.models.signals import post_save
from django.contrib.auth.models import User
from .models import Consumer

@receiver(post_save, sender=User)
def create_consumer_on_user_creation(sender, instance, created, **kwargs):
	print("Signal Recived")
	print(created)
	if created:
		consumer = Consumer.objects.create(user=instance)
		consumer.save()
		print("saved")

