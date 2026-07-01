import os, sys
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
png = os.path.join(script_dir, "logo.png")
ico = os.path.join(script_dir, "logo.ico")

if not os.path.isfile(png):
    print(f"ERROR: {png} not found")
    sys.exit(1)

img = Image.open(png)
img.save(ico, format="ICO", sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)])
print(f"OK: logo.ico created ({os.path.getsize(ico)} bytes)")
