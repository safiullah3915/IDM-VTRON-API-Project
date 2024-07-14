import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from gradio_client import Client, handle_file
import os
import time
import httpx

# Function to select the model image
def select_model_image():
    global model_image_path
    model_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpeg;*.jpg;*.png")])
    if model_image_path:
        img = Image.open(model_image_path)
        img = img.resize((200, 200), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel_model.config(image=img)
        panel_model.image = img

# Function to select the garment image
def select_garment_image():
    global garment_image_path
    garment_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpeg;*.jpg;*.png")])
    if garment_image_path:
        img = Image.open(garment_image_path)
        img = img.resize((200, 200), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel_garment.config(image=img)
        panel_garment.image = img

# Function to process the try-on
def try_on():
    if not model_image_path or not garment_image_path:
        messagebox.showwarning("Warning", "Please select both images.")
        return

    # Display processing message
    processing_label.config(text="Processing...")
    root.update_idletasks()

    # Retry mechanism for API call
    retries = 3
    for attempt in range(retries):
        try:
            # Call the API to process the images
            client = Client("yisol/IDM-VTON")
            result = client.predict(
                dict={"background": handle_file(model_image_path), "layers": [], "composite": None},
                garm_img=handle_file(garment_image_path),
                garment_des="Trying on the shirt",
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )
            break
        except (httpx.ConnectTimeout, httpx.ReadTimeout):
            if attempt < retries - 1:
                time.sleep(2)  # wait for 2 seconds before retrying
            else:
                messagebox.showerror("Error", "Failed to connect to the server. Please try again later.")
                processing_label.config(text="")
                return

    # Load and display the output image on a new screen
    output_image_path = result[0]
    global output_image
    output_image = Image.open(output_image_path)
    output_image = output_image.resize((300, 300), Image.Resampling.LANCZOS)
    output_image = ImageTk.PhotoImage(output_image)
    
    # Create a new window to display the output
    output_window = tk.Toplevel(root)
    output_window.title("Output")
    output_window.geometry("400x500")
    
    panel_output = tk.Label(output_window, image=output_image)
    panel_output.pack(pady=10)
    panel_output.image = output_image
    
    # Buttons below the output image
    save_button = tk.Button(output_window, text="Save", command=save_image)
    save_button.pack(side="left", padx=20, pady=10)
    
    try_other_button = tk.Button(output_window, text="Try Other", command=lambda: [output_window.destroy(), reset_ui()])
    try_other_button.pack(side="right", padx=20, pady=10)
    
    processing_label.config(text="")

# Function to save the output image
def save_image():
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if save_path:
        output_image_path.save(save_path)
        messagebox.showinfo("Info", "Image saved successfully.")

# Function to reset the UI for another try
def reset_ui():
    panel_model.config(image='')
    panel_garment.config(image='')
    global model_image_path, garment_image_path
    model_image_path = ''
    garment_image_path = ''

# Create the main window
root = tk.Tk()
root.title("Virtual Try-On")
root.geometry("400x600")  # Set fixed window size

# Model image panel
panel_model = tk.Label(root)
panel_model.pack(pady=10)
select_model_button = tk.Button(root, text="Select Model Image", command=select_model_image)
select_model_button.pack()

# Garment image panel
panel_garment = tk.Label(root)
panel_garment.pack(pady=10)
select_garment_button = tk.Button(root, text="Select Garment Image", command=select_garment_image)
select_garment_button.pack()

# Try on button
try_on_button = tk.Button(root, text="Try On", command=try_on)
try_on_button.pack(pady=20)

# Processing label
processing_label = tk.Label(root, text="")
processing_label.pack()

# Run the application
root.mainloop()
