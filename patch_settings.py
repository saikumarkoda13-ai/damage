import re

path = 'parcel_damage_classification/settings.py'
content = open(path, 'r', encoding='utf-8').read()

# Replace hard-coded SECRET_KEY
content = re.sub(
    r"SECRET_KEY = '[^']*'",
    "SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-localdev-key')",
    content
)

# Replace DEBUG = True with env-aware version
content = content.replace(
    'DEBUG = True',
    "DEBUG = os.environ.get('DEBUG', 'True') == 'True'"
)

open(path, 'w', encoding='utf-8').write(content)
print('settings.py patched successfully')
