import tkinter as tk
import customtkinter as ctk

from PIL import Image, ImageTk

from authtoken import AuthToken

import torch
from torch import autocast
from diffusers import DiffusionPipeline

app = tk.Tk()
app.geometry('800x600')
app.title('Diffusion Demo')

ctk.set_appearance_mode("dark")

text_box = ctk.CTkEntry(app, width=500, height=40, placeholder_text="Enter a prompt here", font=("Roboto", 20), text_color='black', fg_color='white')
text_box.place(x=10, y=10)

lmain = ctk.CTkLabel(app, width=500, height=500)
lmain.place(x=10, y=110)

model_id ='CompVis/stable-diffusion-v1-4'
device = 'cpu'
pipe = DiffusionPipeline.from_pretrained(model_id)#, torch_dtype=torch.float16)
pipe.to(device)
#pipe.enable_model_cpu_offload()
#pipe.enable_attention_slicing()


def generate():
    with autocast(device):
        image = pipe(text_box.get(), guidance_scale=8.5)['sample'][0]

        image.save('output.png')
        img = ImageTk.PhotoImage(image)
        lmain.config(image=img)
    pass

button = ctk.CTkButton(app, text="Generate", width=120, height=40, font=("Roboto", 20), text_color='white', bg_color='blue', command=generate)
button.place(x=520, y=10)

app.mainloop()