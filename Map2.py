import tkinter as tk
from tkinter import filedialog, simpledialog, colorchooser, messagebox
from PIL import Image, ImageTk
import math

class ImageClickApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Click Coordinates")
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        self.color_button = tk.Button(root, text="Select Color", command=self.select_color)
        self.color_button.pack()

        self.dimensions_button = tk.Button(root, text="Enter Distance and Unit", command=self.enter_dimensions)
        self.dimensions_button.pack()

        self.path_entry_label = tk.Label(root, text="Enter path (comma-separated names):")
        self.path_entry_label.pack()
        self.path_entry = tk.Entry(root, width=50)
        self.path_entry.pack()
        self.path_button = tk.Button(root, text="Draw Path and Calculate Distance", command=self.draw_path)
        self.path_button.pack()

        self.total_distance_label = tk.Label(root, text="Total Distance: 0 m")
        self.total_distance_label.pack()

        self.selected_color = "#FF0000"  # Default color is red
        self.coords_list = []
        self.actual_width = None
        self.actual_height = None
        self.unit = "m"  # Default unit is meters

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
        if file_path:
            self.image = Image.open(file_path)
            self.image = self.image.resize((800, 600), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.canvas.bind("<Button-1>", self.get_coordinates)

    def select_color(self):
        color_code = colorchooser.askcolor(title="Choose color")
        if color_code:
            self.selected_color = color_code[1]

    def enter_dimensions(self):
        dimensions = simpledialog.askstring("Input", "Enter actual width, height and unit (comma-separated, e.g., '10,8,m'):")
        if dimensions:
            dimensions = dimensions.split(',')
            if len(dimensions) >= 2:
                try:
                    self.actual_width = float(dimensions[0])
                    self.actual_height = float(dimensions[1])
                    if len(dimensions) == 3:
                        self.unit = dimensions[2].strip()
                except ValueError:
                    messagebox.showerror("Input Error", "Please enter valid numbers for width and height.")
            else:
                messagebox.showerror("Input Error", "Please enter at least width and height separated by commas.")

    def get_coordinates(self, event):
        x = event.x
        y = event.y
        name = simpledialog.askstring("Input", "Enter name for coordinates:")
        if name:
            self.coords_list.append((x, y, name, self.selected_color))
            self.canvas.create_rectangle(x-5, y-5, x+5, y+5, outline=self.selected_color, fill=self.selected_color)
            self.canvas.create_text(x, y-10, text=name, fill=self.selected_color, font=('Helvetica 10 bold'))
            print(f"Coordinates: ({x}, {y}), Name: {name}, Color: {self.selected_color}")

    def draw_path(self):
        if self.actual_width is None or self.actual_height is None:
            messagebox.showerror("Input Error", "Please enter the actual width, height, and unit of the drawing.")
            return

        path_names = self.path_entry.get().split(',')
        path_coords = []

        for name in path_names:
            for coord in self.coords_list:
                if coord[2] == name.strip():
                    path_coords.append((coord[0], coord[1], coord[3]))

        if len(path_coords) < 2:
            messagebox.showerror("Input Error", "At least two points are needed to draw a path.")
            return

        # Calculate scale factors
        pixel_width = 800
        pixel_height = 600
        scale_x = self.actual_width / pixel_width
        scale_y = self.actual_height / pixel_height

        total_distance = 0

        for i in range(len(path_coords) - 1):
            x1, y1, color1 = path_coords[i]
            x2, y2, color2 = path_coords[i + 1]
            self.canvas.create_line(x1, y1, x2, y2, fill=self.selected_color, width=2, arrow=tk.LAST)
            
            # Calculate distance using scaled values
            dx = (x2 - x1) * scale_x
            dy = (y2 - y1) * scale_y
            distance = math.sqrt(dx**2 + dy**2)
            total_distance += distance

            # Display distance on the line
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            self.canvas.create_text(mid_x, mid_y, text=f"{distance:.2f} {self.unit}", fill=self.selected_color, font=('Helvetica 10 bold'))
            
            print(f"Drew arrow from ({x1}, {y1}) to ({x2}, {y2}) with color {self.selected_color}. Distance: {distance:.2f} {self.unit}")

        self.total_distance_label.config(text=f"Total Distance: {total_distance:.2f} {self.unit}")
        print(f"Total actual distance: {total_distance:.2f} {self.unit}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClickApp(root)
    root.mainloop()