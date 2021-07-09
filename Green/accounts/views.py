from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect
from django.contrib import messages
# Create your views here.
from .forms import NewUserForm
from .models import User

# class ConsumerSignUpView(CreateView):
# 	model = User
# 	form_class = ConsumerSignUpForm
# 	template_name = 'signup_form.html'

# 	def get_context_data(self, **kwargs):
# 		kwargs['user_type'] = 'consumer'
# 		return super().get_context_data(**kwargs)

# 	def form_valid(self, form):
# 		user = form.save()
# 		login(self.request, user)
# 		return redirect('home.html')

# 	def __str__(self):
# 		return self.user

def register_request(request):
	if request.method == "POST":
		form = NewUserForm(request.POST)
		if form.is_valid():
			user = form.save()
			login(request, user)
			return redirect("/")
		print("Not valid")
		messages.error(request, "Unsuccessful registration. Invalid information")
	form = NewUserForm(auto_id=False)
	return render(request, template_name="registration/register.html", context={"form": form})

def login_request(request):
	if request.method == 'POST':
		username = request.POST['username']
		password = request.POST['password']
		user = authenticate(request, username=username, password=password)
		if user is not None:
				login(request, user)
				redirect('/')
		else:
			messages.error(request, 'Invalid Credentials Provided')
	else:
		render(request, 'login.html')


@login_required()
def logout_request(request):
	logout(request)
