from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.sites.shortcuts import get_current_site
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.views import View

from .forms import RegistrationForm, UserEditForm
from .models import UserBase
from .tokens import account_activation_token
from core.views import user_result
from django.utils.decorators import method_decorator


class Dashboard(View):
    @method_decorator(login_required)
    def get(self, request):
        user = UserBase.objects.get(user_name=request.user)
        results = user_result(request)
        length = int(len(results))
        return render(request,
                      'dashboard.html',
                      {'section': 'profile', 'user_': user, 'results': results, 'length': length})


class EditDetails(View):
    @method_decorator(login_required)
    def post(self, request):
        user_form = UserEditForm(instance=request.user, data=request.POST)

        if user_form.is_valid():
            user_form.save()
        return render(request,
                      'edit_details.html', {'user_form': user_form})

    @method_decorator(login_required)
    def get(self, request):
        user_form = UserEditForm(instance=request.user)
        return render(request,
                      'edit_details.html', {'user_form': user_form})


class DeleteUser(View):
    @method_decorator(login_required)
    def get(self, request):
        user = UserBase.objects.get(user_name=request.user)
        user.is_active = False
        user.save()
        logout(request)
        return redirect('/')

class AccountRegister(View):
    def get(self, request):
        if request.user.is_authenticated:
            return redirect('account:dashboard')
        registerForm = RegistrationForm()
        return render(request, 'register.html', {'form': registerForm})

    def post(self, request):
        if request.user.is_authenticated:
            return redirect('account:dashboard')
        registerForm = RegistrationForm(request.POST)
        if registerForm.is_valid():
            user = registerForm.save(commit=False)
            user.email = registerForm.cleaned_data['email']
            user.set_password(registerForm.cleaned_data['password'])
            user.is_active = False
            user.save()
            current_site = get_current_site(request)
            subject = 'Activate your Account'
            message = render_to_string('account_activation_email.html', {
                'user': user,
                'domain': current_site.domain,
                'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                'token': account_activation_token.make_token(user),
            })
            user.email_user(subject=subject, message=message)
            return HttpResponse(f'Account Registered, activation code sent to {user.email}')

class AccountActivate(View):
    def get(self, request, uidb64, token):
        global user
        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = UserBase.objects.get(pk=uid)
        except(TypeError, ValueError, OverflowError, user.DoesNotExist):
            user = None
        if user is not None and account_activation_token.check_token(user, token):
            user.is_active = True
            user.save()
            login(request, user)
            return redirect('account:dashboard')
        else:
            return render(request, 'activation_invalid.html')
