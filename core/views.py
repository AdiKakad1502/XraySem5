import base64
import io
import json
import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm
from django.contrib.auth.decorators import login_required
from .models import results
import io
from django.http import FileResponse
from reportlab.pdfgen import canvas
from datetime import datetime


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


MODEL_PATH = os.path.join(settings.STATIC_ROOT, "New_model.pth")
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


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model_final.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    class_name, human_label = imagenet_mapping[predicted_idx]
    return human_label


@login_required
def index(request):
    image_url = None
    predicted_label = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_url = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)
            try:
                predicted_label = get_prediction(image_bytes)
                user_id = request.user.id
                rng = "rng"
                pn = "pneumonia"
                print(predicted_label)
                if predicted_label == pn:
                    rng = "Infection that inflames air sacs in one or both lungs, which may fill with fluid. With pneumonia, the air sacs may fill with fluid or pus. The infection can be life-threatening to anyone, but particularly to infants, children and people over 65.Symptoms include a cough with phlegm or pus, fever, chills and difficulty breathing."
                else:
                    rng = "Your pneumonia negative! But still remember precaution is better than cure. Wash your hands regularly, especially after you go to the bathroom and before you eat.Eat right, with plenty of fruits and vegetables and remember to Exercise, Get enough sleep, Quit smoking. Stay away from sick people, if possible."
                results.objects.create(user_id=user_id, full_name='name', result=predicted_label, desc=rng)

            except RuntimeError as re:
                print(re)

    else:
        form = ImageUploadForm()

    context = {
        'form': form,
        'image_url': image_url,
        'predicted_label': predicted_label,
    }
    return render(request, 'index.html', context)


@login_required
def user_result(request):
    user_id = request.user.id
    res = results.objects.filter(user_id=user_id)
    return res


@login_required
def GetPDF(request):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    primary_k = request.GET.get('q', 12)
    res = results.objects.get(id=primary_k)
    DateTime = datetime.utcnow().strftime('%Y-%m-%d')
    p.drawString(100, 400, res.result)
    p.drawString(100, 500, "User_name:" + res.user.user_name)
    p.drawString(100, 300, "Date - " + DateTime)

    p.showPage()
    p.save()
    buffer.seek(0)
    return FileResponse(buffer, as_attachment=True, filename=f'test_{primary_k}.pdf')