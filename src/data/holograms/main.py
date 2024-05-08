import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to resize image maintaining aspect ratio
def resize_aspect_fit(image, base=1000):
    w_percent = (base / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)))
    if h_size > base:
        h_percent = (base / float(image.size[1]))
        w_size = int((float(image.size[0]) * float(h_percent)))
        image = image.resize((w_size, base), Image.LANCZOS)
    else:
        image = image.resize((base, h_size), Image.LANCZOS)
    return image

# Load an image using a dialog and resize it
def load_image():
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    return resize_aspect_fit(image)

# Compute FFT of the image and return its magnitude spectrum
def compute_fft(image):
    img_array = np.array(image)
    if img_array.ndim == 3:
        img_array = np.mean(img_array, axis=2)  # Convert to grayscale
    f_transform = np.fft.fftshift(np.fft.fft2(img_array))
    magnitude_spectrum = 20 * np.log(np.abs(f_transform) + 1)  # Add 1 to avoid log(0)
    return f_transform, magnitude_spectrum

# Update the rectangle on canvas
def update_rectangle():
    global rect_id
    canvas.delete(rect_id)
    rect_id = canvas.create_rectangle(x_var.get(), y_var.get(), x_var.get() + width_var.get(), y_var.get() + height_var.get(), outline='red')

# Perform Inverse Fourier Transform on the selected region
def perform_inverse_fourier_transform():
    # Crop the Fourier transform data to the selected region and apply inverse shift
    selected_region = fft_data[y_var.get():y_var.get() + height_var.get(), x_var.get():x_var.get() + width_var.get()]
    ifft_data = np.fft.ifft2(np.fft.ifftshift(selected_region))

    # Display results in a new window
    result_window = tk.Toplevel(root)
    result_window.title("Inverse Fourier Transform Result")
    fig, ax = plt.subplots()
    cax = ax.matshow(np.abs(ifft_data), cmap='gray')
    fig.colorbar(cax)
    canvas = FigureCanvasTkAgg(fig, master=result_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Setup main window
root = tk.Tk()
root.title("Image Fourier Transform")

# Load image and compute its FFT
image = load_image()
fft_data, magnitude_spectrum = compute_fft(image)
magnitude_image = Image.fromarray(np.uint8(magnitude_spectrum))
tk_image = ImageTk.PhotoImage(magnitude_image)

# Create canvas for FFT magnitude display
canvas = tk.Canvas(root, width=magnitude_image.width, height=magnitude_image.height)
canvas.pack()
img_id = canvas.create_image(0, 0, anchor="nw", image=tk_image)

# Draw a rectangle for selection
x_var = tk.IntVar(value=500)
y_var = tk.IntVar(value=450)
width_var = tk.IntVar(value=650)
height_var = tk.IntVar(value=500)
rect_id = canvas.create_rectangle(x_var.get(), y_var.get(), x_var.get() + width_var.get(), y_var.get() + height_var.get(), outline='red')

# Controls for rectangle parameters
control_frame = tk.Frame(root)
control_frame.pack(fill=tk.X)

tk.Label(control_frame, text="X:").pack(side=tk.LEFT)
x_entry = tk.Entry(control_frame, textvariable=x_var)
x_entry.pack(side=tk.LEFT)
x_entry.bind("<Return>", lambda e: update_rectangle())

tk.Label(control_frame, text="Y:").pack(side=tk.LEFT)
y_entry = tk.Entry(control_frame, textvariable=y_var)
y_entry.pack(side=tk.LEFT)
y_entry.bind("<Return>", lambda e: update_rectangle())

tk.Label(control_frame, text="Width:").pack(side=tk.LEFT)
width_entry = tk.Entry(control_frame, textvariable=width_var)
width_entry.pack(side=tk.LEFT)
width_entry.bind("<Return>", lambda e: update_rectangle())

tk.Label(control_frame, text="Height:").pack(side=tk.LEFT)
height_entry = tk.Entry(control_frame, textvariable=height_var)
height_entry.pack(side=tk.LEFT)
height_entry.bind("<Return>", lambda e: update_rectangle())

# Button to perform Inverse Fourier Transform
process_button = tk.Button(root, text="Process", command=perform_inverse_fourier_transform)
process_button.pack()

root.mainloop()
