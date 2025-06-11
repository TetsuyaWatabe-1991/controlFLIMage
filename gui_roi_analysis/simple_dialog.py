import subprocess

def ask_yes_no_cli(prompt="Continue? (y/n): "):
    while True:
        yn = input(prompt)
        if yn.lower() in ['y', 'n']:
            return yn.lower() == 'y'
        print("Invalid input. Please enter 'y' or 'n'.")

def ask_yes_no_gui(prompt="Continue?"):
    dialog_script = f'''
import tkinter as tk
from tkinter import messagebox
root = tk.Tk()
root.withdraw()
result = messagebox.askyesno("Question", "{prompt}")
print("True" if result else "False")
root.destroy()
'''
    result = subprocess.check_output(['python', '-c', dialog_script]).decode().strip()
    return result == "True"



def ask_save_folder_gui(default_folder=""):
    dialog_script = f'import tkinter as tk;from tkinter import filedialog;root=tk.Tk();root.withdraw();print(filedialog.askdirectory(initialdir="{default_folder}"))'
    save_folder = subprocess.check_output(['python', '-c', dialog_script]).decode().strip()
    if save_folder:
        print("save folder:")
        print(save_folder)
        return save_folder

def ask_save_path_gui(filetypes=[("Pickle files","*.pkl")]):
    dialog_script = f'import tkinter as tk;from tkinter import filedialog; root=tk.Tk();root.withdraw();print(filedialog.asksaveasfilename(filetypes={filetypes}))'
    save_path = subprocess.check_output(['python', '-c', dialog_script]).decode().strip()
    if save_path:
        print("save path:")
        print(save_path)
        return save_path
    else:
        print("save path is not defined")
        return None

def ask_open_path_gui(filetypes=[("Pickle files","*.pkl")]):
    dialog_script = f'import tkinter as tk;from tkinter import filedialog;root=tk.Tk();root.withdraw();print(filedialog.askopenfilename(filetypes={filetypes}))'   
    open_path = subprocess.check_output(['python', '-c', dialog_script]).decode().strip()
    if open_path:
        print("open path:")
        print(open_path)
        return open_path
    else:
        print("open path is not defined")
        return None



# デフォルトでCLI版を使用
# ask_yes_no = ask_yes_no_cli

# GUI版に切り替えたい場合は以下のように設定
# ask_yes_no = ask_yes_no_gui 