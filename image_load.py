from PIL import Image
import numpy as np

def image_load():
    jpg = Image.open('dog.jpg')
    array = np.asarray(jpg)
    print(f"Fromat {jpg.format}, size: {jpg.size}, mode: {jpg.mode}")
    print("Shape: ", array.shape)
    print("Image array", array)

if __name__ == '__main__':
    image_load()
