from classifiers import *
from raw_metrics import *
import tkinter as tk
from tkinter import filedialog, messagebox
import warnings

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class NeuralAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Analysis")
        self.root.geometry("500x300")  # Adjusted size to ensure everything fits

        # Instructions
        self.label = tk.Label(
            root,
            text=(
                "Once data has been sorted and verified in Phy, select the session folder "
                "(with the data) and the output folder (project folder where you want all the output files to be)."
            ),
            wraplength=480,
            justify="left",
        )
        self.label.pack(pady=10)

        # Session Folder Selector
        self.session_label = tk.Label(root, text="Session Folder:")
        self.session_label.pack(anchor="w", padx=20, pady=(10, 0))
        self.session_folder = tk.StringVar()
        self.session_entry = tk.Entry(
            root, textvariable=self.session_folder, width=50, state="readonly"
        )
        self.session_entry.pack(anchor="w", padx=20)
        self.session_button = tk.Button(
            root, text="Select", command=self.select_session_folder
        )
        self.session_button.pack(anchor="w", padx=20, pady=5)

        # Output Folder Selector
        self.output_label = tk.Label(root, text="Output Folder:")
        self.output_label.pack(anchor="w", padx=20, pady=(10, 0))
        self.output_folder = tk.StringVar()
        self.output_entry = tk.Entry(
            root, textvariable=self.output_folder, width=50, state="readonly"
        )
        self.output_entry.pack(anchor="w", padx=20)
        self.output_button = tk.Button(
            root, text="Select", command=self.select_output_folder
        )
        self.output_button.pack(anchor="w", padx=20, pady=5)

        # Run Button
        self.run_button = tk.Button(root, text="Run", command=self.run_analysis)
        self.run_button.pack(pady=20)

    def select_session_folder(self):
        folder = filedialog.askdirectory(title="Select Session Folder")
        if folder:
            self.session_folder.set(folder)

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)

    def run_analysis(self):
        session_folder = self.session_folder.get()
        output_folder = self.output_folder.get()
        if not session_folder or not output_folder:
            messagebox.showerror("Error", "Both session and output folders must be selected!")
            return

        # Store folder paths before closing the GUI
        selected_folders = (session_folder, output_folder)

        # Close the GUI
        self.root.destroy()

        # Safely handle folder paths after the GUI is closed
        print(f"Session Folder: {selected_folders[0]}")
        print(f"Output Folder: {selected_folders[1]}")

        # Run the analysis
        animal_output_folder, animal_id, date = generate_metrics(session_folder, output_folder)
        animal_output_folder, animal_id, date = run_classifiers(animal_output_folder, animal_id, date)

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralAnalysisApp(root)
    root.mainloop()