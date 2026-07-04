import urllib.request
import os

files = {
    'age_net.caffemodel': 'https://raw.githubusercontent.com/yu4u/age-gender-estimation/master/models/age_net.caffemodel',
    'gender_net.caffemodel': 'https://raw.githubusercontent.com/yu4u/age-gender-estimation/master/models/gender_net.caffemodel'
}

for filename, url in files.items():
    if os.path.exists(filename):
        print(f'✓ {filename} already exists')
        continue
    try:
        print(f'Downloading {filename}...')
        urllib.request.urlretrieve(url, filename)
        size_mb = os.path.getsize(filename) / (1024**2)
        print(f'✓ Downloaded {filename} ({size_mb:.1f} MB)')
    except Exception as e:
        print(f'✗ Failed: {e}')

print('\nFinal check:')
for f in ['age_net.caffemodel', 'gender_net.caffemodel']:
    if os.path.exists(f):
        print(f'✓ {f} ({os.path.getsize(f)/(1024**2):.1f} MB)')
    else:
        print(f'✗ {f} NOT FOUND')
