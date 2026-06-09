import os
import subprocess
import tkinter as tk
from tkinter import ttk

THIS_FILE = os.path.abspath(__file__)

python_files = []

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in {
        "__pycache__", ".git", ".idea", ".venv", "venv"
    }]

    for file in files:
        if file.endswith(".py"):
            full_path = os.path.abspath(os.path.join(root, file))

            if full_path != THIS_FILE:
                python_files.append(os.path.relpath(full_path))

python_files.sort()


def run_selected():
    selection = listbox.curselection()

    if not selection:
        return

    selected = python_files[selection[0]]

    script_dir = os.path.dirname(os.path.abspath(selected))
    script_name = os.path.basename(selected)

    PYTHON = "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14"

    subprocess.Popen(
        [PYTHON, script_name],
        cwd=script_dir
    )


root = tk.Tk()
root.title("Python Script Launcher")
root.geometry("700x500")

frame = ttk.Frame(root, padding=10)
frame.pack(fill="both", expand=True)

listbox = tk.Listbox(frame, font=("Arial", 12))
listbox.pack(fill="both", expand=True)

for file in python_files:
    listbox.insert(tk.END, file)

run_button = ttk.Button(
    frame,
    text="Run Selected Script",
    command=run_selected
)
run_button.pack(pady=10)

root.mainloop()