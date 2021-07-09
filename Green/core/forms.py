from django import forms
from .models import PredictionModel
class PredictionForm(forms.Form):
	
	image = forms.CharField(widget=forms.TextInput(attrs={'id': 'link', 'hidden': 'hidden'}), label='')

	# def clean_image(self):
	# 	img = self.cleaned_data.get('image')
	# 	if img.image.format in ['JPG', 'JPEG']:
	# 		return img
	# 	else:
	# 		raise forms.ValidationError("Upload a jpg image")
