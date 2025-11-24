
 
import os
import PIL 
import PIL.Image
import pathlib

data_dir = pathlib.Path("C:\\Users\\Owner\\Documents\\Project NDNP\\pathogen").with_suffix("")
_i = 0

img_height = 256
img_width = 256

for file in data_dir.glob("*/*.jpg"):
  print(file)
  with PIL.Image.open(file) as image:
    if image.width == img_width and image.height == img_height:
      continue
  os.remove(file)
  _i += 1
  print(f"File '{file}' deleted successfully.")

    
print(f"{_i} files over 256*256 are removed")