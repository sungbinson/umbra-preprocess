# image_compare_gui.py
import os
from tkinter import Tk, Label, Button, Frame, messagebox
from PIL import Image, ImageTk
import json

class ImageComparer:
    def __init__(self, mask_dir, raster_dir, meta_dir):
        self.mask_dir = mask_dir
        self.raster_dir = raster_dir
        self.meta_dir = meta_dir
        self.files = sorted([f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(raster_dir, f))])
        self.index = 0

        self.root = Tk()
        self.root.title("Image Comparer")

        self.frame = Frame(self.root)
        self.frame.pack()

        self.filename_label = Label(self.root, text="", font=("Helvetica", 32))
        self.filename_label.pack(side="top", pady=10)

        self.mask_label = Label(self.frame)
        self.mask_label.pack(side="left")

        self.raster_label = Label(self.frame)
        self.raster_label.pack(side="right")

        self.coord_label = Label(self.root, text="", font=("Helvetica", 18))
        self.coord_label.pack(side="bottom", pady=10)

        self.delete_button = Button(self.root, text="Delete", command=self.delete_images)
        self.delete_button.pack()

        self.copy_button = Button(self.root, text="Copy Coordinates", command=self.copy_coordinates)
        self.copy_button.pack(side="bottom")

        self.root.bind('<Right>', self.next_image)
        self.root.bind('<Left>', self.prev_image)

        self.load_images()
        self.root.mainloop()

    def load_images(self):
        if self.index < len(self.files):
            mask_path = os.path.join(self.mask_dir, self.files[self.index])
            raster_path = os.path.join(self.raster_dir, self.files[self.index])

            mask_img = Image.open(mask_path)
            raster_img = Image.open(raster_path)

            mask_img = mask_img.resize((800, 800), Image.Resampling.LANCZOS)
            raster_img = raster_img.resize((800, 800), Image.Resampling.LANCZOS)

            self.mask_photo = ImageTk.PhotoImage(mask_img)
            self.raster_photo = ImageTk.PhotoImage(raster_img)

            self.mask_label.config(image=self.mask_photo)
            self.raster_label.config(image=self.raster_photo)

            self.filename_label.config(text=f"{self.files[self.index]} ({self.index + 1}/{len(self.files)})")

            meta_path = os.path.join(self.meta_dir, self.files[self.index])[:63] +'METADATA.json'
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as file:
                    data = json.load(file)
                    collects = data.get("collects", [])
                    coordinates = collects[0]["sceneCenterPointLla"]["coordinates"]
                    if coordinates and len(coordinates) > 1:
                        coord_text = f"{coordinates[1]}, {coordinates[0]}"
                        self.coord_label.config(text=coord_text)
                        self.current_coordinates = coord_text
                    else:
                        self.coord_label.config(text="Coordinates not found")
            else:
                self.coord_label.config(text="Meta file not found")

    def next_image(self, event=None):
        self.index += 1
        if self.index >= len(self.files):
            self.index = 0
        self.load_images()

    def prev_image(self, event=None):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.files) - 1
        self.load_images()

    def delete_images(self):
        mask_path = os.path.join(self.mask_dir, self.files[self.index])
        raster_path = os.path.join(self.raster_dir, self.files[self.index])

        os.remove(mask_path)
        os.remove(raster_path)

        del self.files[self.index]
        if self.index >= len(self.files):
            self.index = 0

        self.load_images()

    def copy_coordinates(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.current_coordinates)
        messagebox.showinfo("Copied", "Coordinates copied to clipboard!")

if __name__ == "__main__":
    mask_directory = "/home/ssb/DATASET/UmbraCoast/masks"
    raster_directory = "/home/ssb/DATASET/UmbraCoast/rasters"
    meta_directory = "/home/ssb/DATASET/UmbraCoast/meta"

    ImageComparer(mask_directory, raster_directory, meta_directory) 