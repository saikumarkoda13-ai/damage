import json
import numpy as np
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from users.models import UserRegistrationModel

# Use tflite_runtime if available, otherwise fallback to tensorflow
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from .models import Prediction
from PIL import Image
from fpdf import FPDF
from django.http import FileResponse
import io
import datetime

# ─── Load TFLite model (Lazy Load to save RAM) ───────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'resnet34_model_quantized.tflite')
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

        # Preprocess with PIL (Memory optimized)
        try:
            img = Image.open(full_path).convert('RGB')
            # Use thumbnail to reduce memory spike before resizing exactly
            if img.width > 512 or img.height > 512:
                img.thumbnail((512, 512))
            img = img.resize((256, 256))
            img_array = np.array(img).astype(np.float32) / 255.0
            print(f"DEBUG: Image shape after processing: {img_array.shape}")
        except Exception as e:
            print(f"ERROR during image preprocessing: {str(e)}")
            return json_response({'success': False, 'message': f'Image preprocessing error: {str(e)}'}, 400)
            
        img_array = np.expand_dims(img_array, axis=0)

        # TFLite Inference (Lazy Load)
        print("DEBUG: Starting TFLite inference...")
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        print(f"DEBUG: Prediction raw result: {prediction}")
        
        # Free memory immediately
        del interpreter
        import gc
        gc.collect()

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
        
        # 3. Save to History
        try:
            Prediction.objects.create(
                image_name=uploaded_file.name,
                prediction=predicted_class,
                confidence=round(confidence * 100, 2),
                severity=severity
            )
        except Exception as db_err:
            print(f"WARNING: Could not save prediction to history: {db_err}")

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


# ─── 7. Dashboard Stats (Admin) ───────────────────────────────────────────────
@csrf_exempt
@require_http_methods(["GET"])
def api_stats(request):
    try:
        total = Prediction.objects.count()
        damaged = Prediction.objects.filter(prediction='Damaged').count()
        intact = Prediction.objects.filter(prediction='Intact').count()
        others = Prediction.objects.filter(prediction='Non-Parcel Image').count()

        # Simple data for a "graph"
        # We'll return percentages
        stats = {
            'total': total,
            'damaged': damaged,
            'intact': intact,
            'others': others,
            'damaged_percent': round((damaged / total * 100), 1) if total > 0 else 0,
            'intact_percent': round((intact / total * 100), 1) if total > 0 else 0,
        }
        return json_response({'success': True, 'stats': stats})
    except Exception as e:
        return json_response({'success': False, 'message': str(e)}, 500)


# ─── 8. Generate PDF Report ──────────────────────────────────────────────────
@csrf_exempt
def api_generate_report(request):
    try:
        # Get data from query params
        prediction = request.GET.get('prediction', 'Unknown')
        confidence = request.GET.get('confidence', '0')
        severity   = request.GET.get('severity', 'Unknown')
        decision   = request.GET.get('decision', 'N/A')
        img_name   = request.GET.get('image', '')

        # 1. Initialize PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 20)
        
        # 2. Header
        pdf.set_text_color(44, 62, 80) # Dark Blue
        pdf.cell(190, 15, "AI PARCEL INSPECTION REPORT", ln=True, align='C')
        pdf.set_font("Helvetica", 'I', 10)
        pdf.cell(190, 10, "Professional Verification & Legal Inspection Proof", ln=True, align='C')
        pdf.ln(5)
        
        # Line break
        pdf.set_draw_color(41, 128, 185)
        pdf.set_line_width(1)
        pdf.line(10, 35, 200, 35)
        pdf.ln(10)

        # 3. Image Section
        if img_name:
            # We assume img_name is just the basename, e.g. "image_123.jpg"
            img_path = os.path.join(settings.MEDIA_ROOT, img_name)
            if os.path.exists(img_path):
                # Center the image
                pdf.image(img_path, x=45, y=pdf.get_y(), w=120)
                pdf.ln(130) # Space after image
            else:
                pdf.set_font("Helvetica", 'I', 10)
                pdf.cell(190, 10, "[Image not found for report]", ln=True, align='C')
                pdf.ln(10)

        # 4. Details Table
        pdf.set_font("Helvetica", 'B', 14)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(190, 10, " INSPECTION SUMMARY", ln=True, fill=True)
        pdf.ln(5)

        pdf.set_font("Helvetica", '', 12)
        pdf.cell(60, 10, "Status:", border=1)
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(130, 10, f" {prediction.upper()}", border=1, ln=True)

        pdf.set_font("Helvetica", '', 12)
        pdf.cell(60, 10, "Confidence Score:", border=1)
        pdf.cell(130, 10, f" {confidence}%", border=1, ln=True)

        pdf.set_font("Helvetica", '', 12)
        pdf.cell(60, 10, "Severity Level:", border=1)
        pdf.set_font("Helvetica", 'B', 12)
        # Color severity based on value
        if severity == 'Severe': pdf.set_text_color(192, 57, 43)
        elif severity == 'Moderate': pdf.set_text_color(243, 156, 18)
        else: pdf.set_text_color(39, 174, 96)
        pdf.cell(130, 10, f" {severity}", border=1, ln=True)
        pdf.set_text_color(0, 0, 0) # Reset color

        pdf.ln(10)

        # 5. Recommendation
        pdf.set_font("Helvetica", 'B', 14)
        pdf.set_fill_color(41, 128, 185)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(190, 12, " RECOMMENDED ACTION", ln=True, fill=True)
        pdf.ln(2)
        pdf.set_text_color(44, 62, 80)
        pdf.set_font("Helvetica", 'B', 16)
        pdf.multi_cell(190, 12, f"\"{decision}\"", align='C')

        # 6. Footer & Legal
        pdf.set_y(-40)
        pdf.set_font("Helvetica", 'I', 8)
        pdf.set_text_color(127, 140, 141)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(190, 10, f"Report Generated on: {now} | System ID: ANTIGRAVITY-AI-EX", ln=True, align='C')
        pdf.multi_cell(190, 5, "Dislaimer: This report is generated by an automated AI computer vision system. It serves as professional inspection proof. Please contact shipping support for formal insurance claims.", align='C')

        # 7. Output to memory buffer
        # In fpdf2, output(dest='S') returns a byte-string (latin-1) or bytes.
        # We'll use the default output() which returns bytes in newer versions or use a custom filename.
        # Actually, let's use output() and wrap in BytesIO.
        pdf_bytes = pdf.output()
        pdf_buffer = io.BytesIO(pdf_bytes)
        pdf_buffer.seek(0)
        
        filename = f"Damage_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return FileResponse(pdf_buffer, as_attachment=True, filename=filename)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"REPORT GENERATION ERROR: {str(e)}")
        return JsonResponse({'success': False, 'message': str(e)}, status=500)
