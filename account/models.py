from django.contrib.auth.models import (AbstractBaseUser, BaseUserManager,
                                        PermissionsMixin)
from django.core.mail import send_mail
from django.db import models
from django.utils.translation import gettext_lazy as _
from django_countries.fields import CountryField


class CustomAccountManager(BaseUserManager):

    def create_superuser(self, email, user_name, password, **other_fields):

        other_fields.setdefault('is_staff', True)
        other_fields.setdefault('is_superuser', True)
        other_fields.setdefault('is_active', True)

        if other_fields.get('is_staff') is not True:
            raise ValueError(
                'Superuser must be assigned to is_staff=True.')
        if other_fields.get('is_superuser') is not True:
            raise ValueError(
                'Superuser must be assigned to is_superuser=True.')

        return self.create_user(email, user_name, password, **other_fields)

    def create_user(self, email, user_name, password, **other_fields):

        if not email:
            raise ValueError(_('You must provide an email address'))

        email = self.normalize_email(email)
        user = self.model(email=email, user_name=user_name,
                          **other_fields)
        user.set_password(password)
        user.save()
        return user


class UserBase(AbstractBaseUser, PermissionsMixin):
    Gender_choices = (
        ('Female', 'Female'),
        ('Male', 'Male'),
        ('Others', 'Others')
    )

    Marital_choices = (
        ('Married', 'Married'),
        ('Not Married', 'Not Married'),
    )

    Blood_group_choices = (
        ('O-positive', 'O-positive'),
        ('O-negative', 'O-negative'),
        ('A-positive', 'A-positive'),
        ('A-negative', 'A-negative'),
        ('B-positive', 'B-positive'),
        ('B-negative', 'B-negative'),
        ('AB-positive', 'AB-positive'),
        ('AB-negative', 'AB-negative')
    )
    email = models.EmailField(_('email address'), unique=True)
    user_name = models.CharField(max_length=150, unique=True)
    first_name = models.CharField(max_length=150, blank=True)
    about = models.TextField(_(
        'about'), max_length=500, blank=True)
    country = CountryField()
    is_active = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    address = models.CharField(blank=True, max_length=200)
    height = models.IntegerField(blank=True, null=True)
    weight = models.IntegerField(blank=True, null=True)
    marital_status = models.CharField(default=False, blank=True, choices=Marital_choices, max_length=25)
    Gender = models.CharField(max_length=15, choices=Gender_choices, blank=True)
    objects = CustomAccountManager()
    Contact = models.IntegerField(null=True, blank=True)
    Blood_group = models.CharField(max_length=15, choices=Blood_group_choices, blank=True)
    ReportCount = models.IntegerField(blank=True, default=0)
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['user_name']

    class Meta:
        verbose_name = "Accounts"
        verbose_name_plural = "Accounts"

    def email_user(self, subject, message):
        send_mail(
            subject,
            message,
            '1@e.com',
            [self.email],
            fail_silently=False,
        )

    def __str__(self):
        return self.user_name
