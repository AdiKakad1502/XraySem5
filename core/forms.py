from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit,Field
from crispy_forms.bootstrap import FormActions


class ImageUploadForm(forms.Form):
    image = forms.ImageField()
