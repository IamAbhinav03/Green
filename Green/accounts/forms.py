from django.forms import widgets
from accounts.models import User
from django import forms
from django.contrib.auth.forms import UserCreationForm


# from .models import Consumer, User

# class ConsumerSignupForm(UserCreationForm):
# 	class Meta(UserCreationForm.Meta):
# 		model = User

# 	@transaction.atomic
# 	def save(self):
# 		user = super().save(commit=False)
# 		user.is_consumer = True
# 		user.save()
# 		Consumer.objects.create(user=user)
# 		return user

class NewUserForm(UserCreationForm):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.fields['email'].widget.attrs.update({'id': 'form2Example11', 'class': 'form-control', 'placeholder': 'Email'})
		self.fields['username'].widget.attrs.update({'id': 'form2Example11', 'class': 'form-control', 'placeholder': 'Username'})
		self.fields['password1'].widget.attrs.update({'id': 'form2Example22', 'class': 'form-control', 'placeholder': 'Password'})
		self.fields['password2'].widget.attrs.update({'id': 'form2Example22', 'class': 'form-control', 'placeholder': 'Confirm Password'})

	email = forms.EmailField(required=True)

	class Meta:
		model = User
		fields = ["email", "username", "password1", "password2"]
		# widgets = {
		# 	"username": forms.CharField(widgets.TextInput(attrs={"id": "form2Example11", "class": "form-control", "placeholder": "Username"})),
		# 	"email": widgets.EmailInput(attrs={"id": "form2Example11", "class": "form-control", "placeholder": "Email"}),
		# 	"password1": widgets.TextInput(attrs={"id": "form2Example22", "class": "form-control", "placeholder": "Password"}),
		# 	"password2": widgets.TextInput(attrs={"id": "form2Example22", "class": "form-control", "placeholder": "Confirm Password"}),
		# }

	def save(self, commit=True):
		user = super(NewUserForm, self).save(commit=False)
		user.email = self.cleaned_data['email']
		if commit:
			user.save()
		return user