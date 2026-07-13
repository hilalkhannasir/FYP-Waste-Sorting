import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import os
from skimage import color


class ImageLabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeling Tool")
        self.root.geometry("900x700")

        self.image_path = None
        self.image = None
        self.image_display = None
        self.original_image = None
        self.zoom_factor = 1.0
        self.clicks = []
        self.labels_data = []
        self.class_map = {
            "cardboard":0,
            "cloth":1,
            "egg-shell":2,
            "organic":3,
            "plastic":4,
            "plastic-bag":5,
            "tea-waste":6,
            "background":7
        }

        self.csv_path = tk.StringVar(value="labeled_data.csv")
        self.append_mode = tk.BooleanVar(value=True)

        self.setup_ui()

    def setup_ui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        tk.Button(control_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="CSV Path:").pack(side=tk.LEFT, padx=5)
        tk.Entry(control_frame, textvariable=self.csv_path, width=30).pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Select Class:").pack(side=tk.LEFT, padx=5)
        self.class_var = tk.StringVar(value="cardboard")
        self.class_dropdown = ttk.Combobox(control_frame, textvariable=self.class_var,
                                values=[ "cardboard","cloth","egg-shell","organic", "plastic","plastic-bag","tea-waste","background"], width=10)
        self.class_dropdown.pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="Save Data", command=self.save_data).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Undo", command=self.undo_last_selection).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(control_frame, text="Append to CSV", variable=self.append_mode).pack(side=tk.LEFT, padx=5)

        self.image_frame = tk.Frame(self.root, bg='gray')
        self.image_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.canvas = tk.Canvas(self.image_frame, bg='gray')
        self.canvas.pack(expand=True, fill=tk.BOTH)

        self.canvas.bind("<Button-1>", self.on_image_click)

        self.status_label = tk.Label(self.root, text="Load an image to start labeling", font=('Arial', 10))
        self.status_label.pack(pady=5)

        self.info_label = tk.Label(self.root, text="Click on image to select pixels (3x3 window)", font=('Arial', 9))
        self.info_label.pack(pady=2)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )

        if file_path:
            try:
                self.image_path = file_path
                self.image = Image.open(file_path)
                self.original_image = self.image.copy()

                img_array = np.array(self.image.convert('RGB'))

                self.lab_image = color.rgb2lab(img_array)

                self.display_image()

                self.clicks = []
                self.labels_data = []
                self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                self.info_label.config(text="Click on image to select pixels (3x3 window)")

            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")

    def display_image(self):
        if self.image is None:
            return

        self.canvas.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600

        img_width, img_height = self.image.size
        scale_w = canvas_width / img_width
        scale_h = canvas_height / img_height
        scale = min(scale_w, scale_h, 1.0)

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        if new_width > 0 and new_height > 0:
            self.display_img = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(self.display_img)

            self.canvas.delete("all")
            x_center = (canvas_width - new_width) // 2
            y_center = (canvas_height - new_height) // 2
            self.canvas.create_image(x_center, y_center, anchor=tk.NW, image=self.photo)
            self.canvas.image = self.photo

            self.scale = scale
            self.x_offset = x_center
            self.y_offset = y_center
            self.display_width = new_width
            self.display_height = new_height

            self.redraw_selection_dots()

    def redraw_selection_dots(self):
        for orig_x, orig_y in self.clicks:
            disp_x = int(orig_x * self.scale + self.x_offset)
            disp_y = int(orig_y * self.scale + self.y_offset)

            self.canvas.create_oval(disp_x - 3, disp_y - 3, disp_x + 3, disp_y + 3,
                                    fill='red', outline='red')

    def on_image_click(self, event):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        x = event.x - self.x_offset
        y = event.y - self.y_offset

        if 0 <= x < self.display_width and 0 <= y < self.display_height:
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)

            self.clicks.append((orig_x, orig_y))
            self.extract_lab_values(orig_x, orig_y)

            self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3,
                                    fill='red', outline='red')

            self.status_label.config(text=f"Clicked at ({orig_x}, {orig_y}) - Total selections: {len(self.clicks)}")
            self.info_label.config(text=f"Selected pixel ({orig_x}, {orig_y}) with class {self.class_var.get()}")

    def extract_lab_values(self, x, y):
        height, width, _ = self.lab_image.shape
        half_window = 1

        x_start = max(0, x - half_window)
        x_end = min(width, x + half_window + 1)
        y_start = max(0, y - half_window)
        y_end = min(height, y + half_window + 1)
        window = self.lab_image[y_start:y_end, x_start:x_end]

        lab_values = window.reshape(-1, 3).tolist()
        lab_values_rounded = [[round(val[0], 2), round(val[1], 2), round(val[2], 2)] for val in lab_values]

        class_label = self.class_map[self.class_var.get()]

        self.labels_data.append({
            'pixels': lab_values_rounded,
            'class': class_label,
            'position': (x, y)
        })

        print(f"Selected window at ({x}, {y}) with {len(lab_values_rounded)} pixels, class: {class_label}")
        if lab_values_rounded:
            print(f"Sample L,a,b: {lab_values_rounded[0]}")  # L will be 0-100, a,b will be -128 to 127

    def undo_last_selection(self):
        if not self.labels_data:
            messagebox.showinfo("Info", "No selections to undo")
            return

        self.labels_data.pop()
        if self.clicks:
            self.clicks.pop()
        self.display_image()

        self.status_label.config(text=f"Undo successful. Remaining selections: {len(self.clicks)}")
        if self.clicks:
            self.info_label.config(
                text=f"Last selection at ({self.clicks[-1][0]}, {self.clicks[-1][1]}) with class {self.labels_data[-1]['class'] if self.labels_data else 'N/A'}")
        else:
            self.info_label.config(text="No selections remaining")

    def save_data(self):
        if not self.labels_data:
            messagebox.showwarning("Warning", "No data to save. Please select some pixels first.")
            return

        try:
            rows = []
            for data in self.labels_data:
                row = []
                for pixel in data['pixels']:
                    row.extend(pixel)
                row.append(data['class'])
                rows.append(row)

            max_pixels = max(len(data['pixels']) for data in self.labels_data)
            columns = []
            for i in range(max_pixels):
                columns.extend([f'L{i + 1}', f'a{i + 1}', f'b{i + 1}'])
            columns.append('class')

            padded_rows = []
            for row in rows:
                while len(row) < len(columns):
                    row.append('')
                padded_rows.append(row)

            df_new = pd.DataFrame(padded_rows, columns=columns)

            csv_path = self.csv_path.get()
            if os.path.exists(csv_path) and self.append_mode.get():
                # Read existing data
                df_existing = pd.read_csv(csv_path)
                # Append new data
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(csv_path, index=False)
                messagebox.showinfo("Success", f"Appended {len(self.labels_data)} new selections to {csv_path}")
                self.status_label.config(text=f"Appended {len(self.labels_data)} selections to {csv_path}")
            else:
                df_new.to_csv(csv_path, index=False)
                if os.path.exists(csv_path):
                    messagebox.showinfo("Success",
                                        f"Overwrote file with {len(self.labels_data)} selections at {csv_path}")
                else:
                    messagebox.showinfo("Success",
                                        f"Created new file with {len(self.labels_data)} selections at {csv_path}")
                self.status_label.config(text=f"Saved {len(self.labels_data)} selections to {csv_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not save data: {str(e)}")

    def clear_all(self):
        self.clicks = []
        self.labels_data = []
        self.canvas.delete("all")
        self.status_label.config(text="Cleared all selections")
        self.info_label.config(text="Click on image to select pixels (3x3 window)")

        if self.image is not None:
            self.display_image()


def main():
    root = tk.Tk()
    ImageLabelingTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()