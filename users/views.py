import json
from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
from django.core.files.storage import FileSystemStorage
import os


# Use tflite_runtime if available, otherwise fallback to tensorflow
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from PIL import Image
import numpy as np
import os
from django.shortcuts import render, HttpResponse
from django.conf import settings



# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})





import numpy as np
import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Use tflite_runtime if available, otherwise fallback to tensorflow
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from PIL import Image

# Build the absolute path to the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'resnet34_model_quantized.tflite')

# Load TFLite model once
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = ['Damaged', 'Intact']

# Prediction view
def predict_view(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_path = fs.path(file_path)

        # Preprocess image with PIL
        img = Image.open(full_path).resize((256, 256))
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Ensure 3 channels
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
            
        img_array = np.expand_dims(img_array, axis=0)

        # Predict with TFLite
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        if len(prediction) == 1:  # sigmoid model
            prob = float(prediction[0])

            if prob >= 0.85:
                predicted_class = "Intact"
                confidence = prob
            elif prob <= 0.15:
                predicted_class = "Damaged"
                confidence = 1 - prob
            else:
                predicted_class = "Non-Parcel Image"
                confidence = prob if prob > 0.5 else (1 - prob)

        else:  # softmax model
            confidence = float(np.max(prediction))
            predicted_class = class_names[int(np.argmax(prediction))]

            if confidence < 0.60:
                predicted_class = "Non-Parcel Image"

        context = {
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}",
            'image_url': fs.url(file_path),
        }

    return render(request, 'users/predict.html', context)
    