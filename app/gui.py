
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import requests
import base64
import io

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Multi-Model YOLO Client")
        self.geometry("1200x700")

        # The base URL of your FastAPI server
        self.api_base_url = "http://localhost:8000"  # Adjust if needed

        # Models we can choose from
        self.model_names = ["Large_without_aug","Large_aug","medium_no_aug","medium_with_aug"]

        # Data structures
        self.model_frames = {}
        self.model_file_paths = {}
        self.model_images_tk = {}  # store PhotoImage references
        self.defect_textbox = None

        # Layout: left (sidebar), center (frames), right (defect box)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=4)
        self.grid_columnconfigure(2, weight=2)

        # Left sidebar
        self.left_sidebar = customtkinter.CTkFrame(self, corner_radius=0)
        self.left_sidebar.grid(row=0, column=0, sticky="nsew")

        # Create a button for each model
        for i, model_name in enumerate(self.model_names):
            button = customtkinter.CTkButton(
                self.left_sidebar,
                text=model_name,
                command=lambda mn=model_name: self.show_model(mn)
            )
            button.grid(row=i, column=0, padx=20, pady=(20 if i == 0 else 10), sticky="ew")

        # Theme selector
        self.appearance_mode_menu = customtkinter.CTkOptionMenu(
            self.left_sidebar,
            values=["Light", "Dark", "System"],
            command=customtkinter.set_appearance_mode
        )
        self.appearance_mode_menu.grid(
            row=len(self.model_names), column=0, padx=20, pady=20, sticky="s"
        )

        # Center frames (one per model)
        self.active_frame = None
        for model_name in self.model_names:
            frame = self.create_model_frame(model_name)
            self.model_frames[model_name] = frame
            self.model_file_paths[model_name] = None
            self.model_images_tk[model_name] = None

        # Show first model by default
        self.show_model(self.model_names[0])

        # Right sidebar: defect display
        self.right_sidebar = customtkinter.CTkFrame(self, corner_radius=0)
        self.right_sidebar.grid(row=0, column=2, sticky="nsew", padx=(5, 0))

        defect_label = customtkinter.CTkLabel(
            self.right_sidebar, text="Class Counts (Defects)", font=("Arial", 16, "bold")
        )
        defect_label.pack(padx=20, pady=(20, 5))

        self.defect_textbox = customtkinter.CTkTextbox(
            self.right_sidebar, width=300, height=500, font=("Arial", 14)
        )
        self.defect_textbox.pack(padx=20, pady=10, fill="both", expand=True)
        self.update_defects_text(["No defects yet."])

    def create_model_frame(self, model_name: str):
        frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)

        title_label = customtkinter.CTkLabel(
            frame,
            text=f"{model_name} Frame",
            font=("Arial", 18, "bold")
        )
        title_label.grid(row=0, column=0, padx=20, pady=(20, 5))

        file_label = customtkinter.CTkLabel(frame, text="No file uploaded.", wraplength=600)
        file_label.grid(row=1, column=0, padx=20, pady=10)

        image_label = customtkinter.CTkLabel(frame, text="")
        image_label.grid(row=2, column=0, padx=20, pady=10)

        # Buttons row
        btn_frame = customtkinter.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=3, column=0, pady=10)

        upload_btn = customtkinter.CTkButton(
            btn_frame,
            text="Upload",
            command=lambda: self.upload_file(model_name, file_label, image_label)
        )
        upload_btn.pack(side="left", padx=(0, 20))

        process_btn = customtkinter.CTkButton(
            btn_frame,
            text="Process",
            command=lambda: self.process_file(model_name, file_label, image_label)
        )
        process_btn.pack(side="left", padx=(0, 20))

        reset_btn = customtkinter.CTkButton(
            btn_frame,
            text="Reset",
            command=lambda: self.reset_model(model_name, file_label, image_label)
        )
        reset_btn.pack(side="left")

        return frame

    def show_model(self, model_name: str):
        if self.active_frame:
            self.active_frame.grid_forget()
        new_frame = self.model_frames[model_name]
        new_frame.grid(row=0, column=1, sticky="nsew")
        self.active_frame = new_frame

    def upload_file(self, model_name: str, file_label, image_label):
        """Select a local image file and display in the label."""
        path = filedialog.askopenfilename(
            title=f"Select File for {model_name}",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All Files", "*.*")]
        )
        if path:
            file_label.configure(text=f"{model_name} file:\n{path}")
            self.model_file_paths[model_name] = path

            # Display the uploaded image (thumbnail in GUI)
            pil_img = Image.open(path)
            tk_img = self.resize_pil_to_tk(pil_img, 640)
            image_label.configure(image=tk_img, text="")
            self.model_images_tk[model_name] = tk_img
        else:
            file_label.configure(text="No file uploaded.")
            image_label.configure(image="", text="")
            self.model_file_paths[model_name] = None
            self.model_images_tk[model_name] = None

    def process_file(self, model_name: str, file_label, image_label):
        """Send file to API, get annotated image + class counts back."""
        path = self.model_file_paths[model_name]
        if not path:
            self.update_defects_text(["No file to process."])
            return

        # 1) Send request to server
        files = {"file": open(path, "rb")}
        url = f"{self.api_base_url}/predict/{model_name}"
        try:
            response = requests.post(url, files=files)
        except Exception as e:
            self.update_defects_text([f"Error contacting server: {e}"])
            return

        if response.status_code != 200:
            self.update_defects_text([f"Error from API: {response.text}"])
            return

        data = response.json()
        if "error" in data:
            # The API returned an error
            self.update_defects_text([f"API Error: {data['error']}"])
            return
        # 2) Display class counts
        class_counts = data["class_counts"]
        if not class_counts:
            self.update_defects_text(["No objects detected."])
        else:
            # Convert dict to lines
            lines = [f"{k}: {v}" for k, v in class_counts.items()]
            self.update_defects_text(lines)

        # 3) Decode annotated image
        b64_img = data["annotated_image_base64"]
        annotated_pil = self.decode_base64_to_pil(b64_img)
        annotated_tk = self.resize_pil_to_tk(annotated_pil, 640)

        # 4) Display annotated image
        image_label.configure(image=annotated_tk, text="")
        self.model_images_tk[model_name] = annotated_tk

    def reset_model(self, model_name: str, file_label, image_label):
        """Clear everything for this model."""
        self.model_file_paths[model_name] = None
        self.model_images_tk[model_name] = None

        file_label.configure(text="No file uploaded.")
        image_label.configure(image="", text="")
        self.update_defects_text(["Reset complete."])

    def update_defects_text(self, lines):
        """Show lines in the defect_textbox on the right sidebar."""
        self.defect_textbox.delete("0.0", "end")
        if not lines:
            lines = ["No defects."]
        for line in lines:
            self.defect_textbox.insert("end", line + "\n")

    def resize_pil_to_tk(self, pil_img, width):
        """Resize PIL image to the given width, keep aspect, return Tk image."""
        w, h = pil_img.size
        scale = width / float(w)
        new_h = int(h * scale)
        pil_img = pil_img.resize((width, new_h), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil_img)

    def decode_base64_to_pil(self, b64_str):
        """Decode a base64 string into a PIL image."""
        img_data = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(img_data))

if __name__ == "__main__":
    customtkinter.set_appearance_mode("System")
    customtkinter.set_default_color_theme("blue")

    app = App()
    app.mainloop()
