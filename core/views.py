import base64
import json
import os
import torch
import torch.nn as nn
from django.utils.decorators import method_decorator
from django.views import View
from torchvision import models
from torchvision import transforms
from PIL import Image
from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm
from django.contrib.auth.decorators import login_required
from .models import results
import io
from django.http import HttpResponse
from django.template.loader import get_template
from xhtml2pdf import pisa


def CNN_Model(pretrained=True):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model = models.densenet121(pretrained=pretrained)
    num_filters = model.classifier.in_features
    model.classifier = nn.Linear(num_filters, 2)
    model = model.to(device)
    return model

MODEL_PATH_Covid = os.path.join(settings.STATIC_ROOT, "Covid-normal-differentiator.pth")
model_final_Covid = CNN_Model(pretrained=False)
model_final_Covid.to(torch.device('cpu'))
model_final_Covid.load_state_dict(torch.load(MODEL_PATH_Covid, map_location='cpu'))
model_final_Covid.eval()
json_path_covid = os.path.join(settings.STATIC_ROOT, "classes-covid.json")
imagenet_mapping_Covid = json.load(open(json_path_covid))

MODEL_PATH_XRAY = os.path.join(settings.STATIC_ROOT, "Xray-normal-differentiator .pth")
model_final_xray = CNN_Model(pretrained=False)
model_final_xray.to(torch.device('cpu'))
model_final_xray.load_state_dict(torch.load(MODEL_PATH_XRAY, map_location='cpu'))
model_final_xray.eval()
json_path1 = os.path.join(settings.STATIC_ROOT, "classes-xray.json")
imagenet_mapping_Xray = json.load(open(json_path1))

MODEL_PATH = os.path.join(settings.STATIC_ROOT, "Pneumonia-normal-differentiator.pth")
model_final = CNN_Model(pretrained=False)
model_final.to(torch.device('cpu'))
model_final.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model_final.eval()
json_path = os.path.join(settings.STATIC_ROOT, "classes.json")
imagenet_mapping = json.load(open(json_path))
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_nums, std=std_nums)
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return my_transforms(image).unsqueeze(0)


def get_prediction_Covid(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model_final_Covid.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    class_name, human_label = imagenet_mapping_Covid[predicted_idx]
    return human_label


def get_prediction_Xray(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model_final_xray.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    class_name, human_label = imagenet_mapping_Xray[predicted_idx]
    return human_label


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model_final.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    class_name, human_label = imagenet_mapping[predicted_idx]
    return human_label

class MainPage(View):
    @method_decorator(login_required)
    def post(self, request):
        image_url = None
        predicted_label = None
        check_label = "normal"
        predicted_label_covid = None
        pre = "Not a valid image!"
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_url = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)
            try:
                check_for_xray = get_prediction_Xray(image_bytes)
                if check_for_xray == check_label:
                    predicted_label = get_prediction(image_bytes)
                    user_id = request.user.id
                    predicted_label_covid = get_prediction_Covid(image_bytes)
                    description = f"Covid-19 : {predicted_label_covid}\nPneumonia : {predicted_label}"
                    results.objects.create(user_id=user_id, full_name='name', result_pneumonia=predicted_label,
                                           result_covid=predicted_label_covid,
                                           desc=description, image=image)
            except RuntimeError as re:
                print(re)
        context = {
            'image_url': image_url,
            'form': form,
            'predicted_label': predicted_label,
            'predicted_label_covid': predicted_label_covid,
            'pre': pre,
        }
        return render(request, 'index.html', context)
    @method_decorator(login_required)
    def get(self, request):
        form = ImageUploadForm()
        image_url = None
        predicted_label = None
        predicted_label_covid = None
        pre = "Not a valid image!"
        context = {
            'image_url': image_url,
            'form': form,
            'predicted_label': predicted_label,
            'predicted_label_covid': predicted_label_covid,
            'pre': pre,
        }
        return render(request, 'index.html', context)
@login_required
def user_result(request):
    user_id = request.user.id
    res = results.objects.filter(user_id=user_id)
    return res

class MakePDF(View):
    @method_decorator(login_required)
    def get(self, request):
        primary_k = request.GET.get('q')
        res = results.objects.get(id=primary_k)
        if request.user == res.user:
            template_path = 'pdfFile1.html'
            context = {'res': res}
            response = HttpResponse(content_type='application/pdf')
            response['Content-Disposition'] = 'attachment; filename="report.pdf"'
            template = get_template(template_path)
            html = template.render(context)
            pisa_status = pisa.CreatePDF(
                html, dest=response
            )
            if pisa_status.err:
                return HttpResponse("Something went wrong!<pre>" + html + '</pre>')
            return response
        return HttpResponse("That's not your report bud! Not cool!")