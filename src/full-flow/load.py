from PIL import Image
import numpy as np

def load_hologram(file_path):
    hologram = Image.open(file_path)
    hologram = np.array(hologram.convert('L'), dtype=np.float32)

    max_dim = max(hologram.shape)
    
    pad_width = ((0, max_dim - hologram.shape[0]), (0, max_dim - hologram.shape[1]))
    
    hologram = np.pad(hologram, pad_width, mode='constant')

    return hologram