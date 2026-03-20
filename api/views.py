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
        print(f"DEBUG: api_predict received request. Method: {request.method}")
        
        if not request.FILES.get('image'):
            print("ERROR: No image file found in request.FILES")
            return json_response({'success': False, 'message': 'No image file provided.'}, 400)

        uploaded_file = request.FILES['image']
        print(f"DEBUG: Uploaded file name: {uploaded_file.name}, Size: {uploaded_file.size} bytes")
        
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_path = fs.path(file_path)
        print(f"DEBUG: File saved at: {full_path}")

        # Preprocess with PIL
        try:
            img = Image.open(full_path).convert('RGB').resize((256, 256))
            img_array = np.array(img).astype(np.float32) / 255.0
            print(f"DEBUG: Image shape after processing: {img_array.shape}")
        except Exception as e:
            print(f"ERROR during image preprocessing: {str(e)}")
            return json_response({'success': False, 'message': f'Image preprocessing error: {str(e)}'}, 400)
            
        img_array = np.expand_dims(img_array, axis=0)

        # TFLite Inference
        print("DEBUG: Starting TFLite inference...")
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        print(f"DEBUG: Prediction raw result: {prediction}")

        # 1. First, determine predicted_class and confidence
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
            prob = float(np.max(prediction)) # treat max as prob for severity calc
            confidence = float(np.max(prediction))
            predicted_class = class_names[int(np.argmax(prediction))]
            if confidence < 0.60:
                predicted_class = 'Non-Parcel Image'

        # 2. Final categorization and severity mapping
        severity = 'N/A'
        decision = 'N/A'
        color = '#95a5a6'  # Default gray

        if predicted_class == 'Intact':
            severity = 'Safe'
            decision = 'Deliver normally'
            color = '#2ecc71'  # Green
        elif predicted_class == 'Damaged':
            # Calculate damage confidence (1 - prob if prob is low, or direct prob if softmax)
            damage_conf = (1 - prob) if len(prediction) == 1 else confidence
            
            if damage_conf >= 0.60:
                severity = 'Severe'
                decision = 'Reject / Return parcel'
                color = '#e74c3c'  # Red
            else:
                severity = 'Moderate'
                decision = 'Handle carefully'
                color = '#f1c40f'  # Yellow
        elif predicted_class == 'Non-Parcel Image':
            severity = 'Unknown'
            decision = 'N/A'
            color = '#95a5a6'

        print(f"DEBUG: Final prediction: {predicted_class} | Severity: {severity} | Decision: {decision}")
        
        return json_response({
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence * 100, 2),
            'severity': severity,
            'decision': decision,
            'color': color,
            'image_url': fs.url(file_path),
        })
    except Exception as e:
        print(f"CRITICAL ERROR in api_predict: {str(e)}")
        import traceback
        traceback.print_exc()
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
        # Try JSON body first
        try:
            data = json.loads(request.body)
            uid = data.get('uid')
        except:
            uid = None

        # Fallback to Form POST if JSON failed or uid missing
        if not uid:
            uid = request.POST.get('uid')

        if not uid:
            print("ERROR: User ID missing in both JSON body and POST data")
            return json_response({'success': False, 'message': 'User ID is required.'}, 400)

        # Cast uid to int for reliability
        try:
            uid_int = int(uid)
        except (ValueError, TypeError):
            print(f"ERROR: Invalid User ID format: {uid}")
            return json_response({'success': False, 'message': 'Invalid User ID format.'}, 400)

        print(f"DEBUG: Attempting to activate user ID: {uid_int}")
        
        updated = UserRegistrationModel.objects.filter(id=uid_int).update(status='activated')
        
        if updated:
            print(f"DEBUG: Successfully activated user ID: {uid_int}")
            return json_response({'success': True, 'message': f'User {uid_int} activated successfully.'})
            
        print(f"DEBUG: User ID {uid_int} not found in database")
        return json_response({'success': False, 'message': 'User not found.'}, 404)
    except Exception as e:
        print(f"DEBUG: Exception in api_activate_user: {str(e)}")
        return json_response({'success': False, 'message': str(e)}, 500)
