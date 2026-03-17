import json
import numpy as np
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import FileSystemStorage
from users.models import UserRegistrationModel

# Use tflite_runtime if available, otherwise fallback to tensorflow
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from PIL import Image

# ─── Load TFLite model ────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'resnet34_model_quantized.tflite')
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = ['Damaged', 'Intact']


# ─── Helper ───────────────────────────────────────────────────────────────────
def json_response(data, status=200):
    return JsonResponse(data, status=status, safe=isinstance(data, dict))


# ─── 1. User Registration ─────────────────────────────────────────────────────
@csrf_exempt
@require_http_methods(["POST"])
def api_register(request):
    try:
        data = json.loads(request.body)
        name     = data.get('name', '').strip()
        loginid  = data.get('loginid', '').strip()
        password = data.get('password', '').strip()
        mobile   = data.get('mobile', '').strip()
        email    = data.get('email', '').strip()
        locality = data.get('locality', '').strip()
        address  = data.get('address', '').strip()
        city     = data.get('city', '').strip()
        state    = data.get('state', '').strip()

        if not all([name, loginid, password, mobile, email]):
            return json_response({'success': False, 'message': 'All required fields must be filled.'}, 400)

        if UserRegistrationModel.objects.filter(loginid=loginid).exists():
            return json_response({'success': False, 'message': 'Login ID already exists.'}, 400)
        if UserRegistrationModel.objects.filter(email=email).exists():
            return json_response({'success': False, 'message': 'Email already exists.'}, 400)
        if UserRegistrationModel.objects.filter(mobile=mobile).exists():
            return json_response({'success': False, 'message': 'Mobile number already exists.'}, 400)

        user = UserRegistrationModel.objects.create(
            name=name, loginid=loginid, password=password,
            mobile=mobile, email=email, locality=locality,
            address=address, city=city, state=state, status='pending'
        )
        return json_response({'success': True, 'message': 'Registration successful. Please wait for admin activation.'})
    except Exception as e:
        return json_response({'success': False, 'message': str(e)}, 500)


# ─── 2. User Login ────────────────────────────────────────────────────────────
@csrf_exempt
@require_http_methods(["POST"])
def api_login(request):
    try:
        data     = json.loads(request.body)
        loginid  = data.get('loginid', '').strip()
        password = data.get('password', '').strip()

        user = UserRegistrationModel.objects.get(loginid=loginid, password=password)
        if user.status != 'activated':
            return json_response({'success': False, 'message': 'Your account is not yet activated by admin.'}, 403)

        return json_response({
            'success': True,
            'message': 'Login successful.',
            'user': {
                'id': user.id,
                'name': user.name,
                'loginid': user.loginid,
                'email': user.email,
                'mobile': user.mobile,
            }
        })
    except UserRegistrationModel.DoesNotExist:
        return json_response({'success': False, 'message': 'Invalid login ID or password.'}, 401)
    except Exception as e:
        return json_response({'success': False, 'message': str(e)}, 500)


# ─── 3. Predict (image upload) ────────────────────────────────────────────────
@csrf_exempt
@require_http_methods(["POST"])
def api_predict(request):
    try:
        if not request.FILES.get('image'):
            return json_response({'success': False, 'message': 'No image file provided.'}, 400)

        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_path = fs.path(file_path)

        # Preprocess with PIL
        img = Image.open(full_path).resize((256, 256))
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Ensure 3 channels
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
            
        img_array = np.expand_dims(img_array, axis=0)

        # TFLite Inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        if len(prediction) == 1:  # sigmoid
            prob = float(prediction[0])
            if prob >= 0.85:
                predicted_class = 'Intact'
                confidence = prob
            elif prob <= 0.15:
                predicted_class = 'Damaged'
                confidence = 1 - prob
            else:
                predicted_class = 'Non-Parcel Image'
                confidence = prob if prob > 0.5 else (1 - prob)
        else:  # softmax
            confidence = float(np.max(prediction))
            predicted_class = class_names[int(np.argmax(prediction))]
            if confidence < 0.60:
                predicted_class = 'Non-Parcel Image'

        return json_response({
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence * 100, 2),
            'image_url': fs.url(file_path),
        })
    except Exception as e:
        return json_response({'success': False, 'message': str(e)}, 500)


# ─── 4. Admin Login ───────────────────────────────────────────────────────────
@csrf_exempt
@require_http_methods(["POST"])
def api_admin_login(request):
    try:
        data    = json.loads(request.body)
        loginid = data.get('loginid', '').strip()
        password = data.get('password', '').strip()

        if loginid.lower() == 'admin' and password.lower() == 'admin':
            return json_response({'success': True, 'message': 'Admin login successful.'})
        return json_response({'success': False, 'message': 'Invalid admin credentials.'}, 401)
    except Exception as e:
        return json_response({'success': False, 'message': str(e)}, 500)


# ─── 5. View Registered Users (Admin) ─────────────────────────────────────────
@csrf_exempt
@require_http_methods(["GET"])
def api_users(request):
    try:
        users = UserRegistrationModel.objects.all().values(
            'id', 'name', 'loginid', 'email', 'mobile', 'locality', 'address', 'city', 'state', 'status'
        )
        return json_response({'success': True, 'users': list(users)})
    except Exception as e:
        return json_response({'success': False, 'message': str(e)}, 500)


# ─── 6. Activate User (Admin) ─────────────────────────────────────────────────
@csrf_exempt
@require_http_methods(["POST"])
def api_activate_user(request):
    try:
        data = json.loads(request.body)
        uid  = data.get('uid')
        if not uid:
            return json_response({'success': False, 'message': 'User ID is required.'}, 400)

        updated = UserRegistrationModel.objects.filter(id=uid).update(status='activated')
        if updated:
            return json_response({'success': True, 'message': f'User {uid} activated successfully.'})
        return json_response({'success': False, 'message': 'User not found.'}, 404)
    except Exception as e:
        return json_response({'success': False, 'message': str(e)}, 500)
