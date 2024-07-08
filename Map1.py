import tkinter as tk
from tkinter import filedialog, simpledialog, colorchooser
from PIL import Image, ImageTk

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

        self.path_entry_label = tk.Label(root, text="Enter path (comma-separated names):")
        self.path_entry_label.pack()
        self.path_entry = tk.Entry(root, width=50)
        self.path_entry.pack()
        self.path_button = tk.Button(root, text="Draw Path", command=self.draw_path)
        self.path_button.pack()

        self.selected_color = "#FF0000"  # Default color is red
        self.coords_list = []

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
        path_names = self.path_entry.get().split(',')
        path_coords = []

        for name in path_names:
            for coord in self.coords_list:
                if coord[2] == name.strip():
                    path_coords.append((coord[0], coord[1], coord[3]))

        if len(path_coords) < 2:
            print("At least two points are needed to draw a path.")
            return

        for i in range(len(path_coords) - 1):
            x1, y1, color1 = path_coords[i]
            x2, y2, color2 = path_coords[i + 1]
            self.canvas.create_line(x1, y1, x2, y2, fill=self.selected_color, width=2, arrow=tk.LAST)
            print(f"Drew arrow from ({x1}, {y1}) to ({x2}, {y2}) with color {self.selected_color}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClickApp(root)
    root.mainloop()
