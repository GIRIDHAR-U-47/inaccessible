import os
import subprocess
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

class ScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Antivirus Scanner")
        
        # Create UI components
        self.create_widgets()

    def create_widgets(self):
        # Frame for buttons
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons
        self.scan_files_button = ttk.Button(self.frame, text="Select Files to Scan", command=self.select_files)
        self.scan_files_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.scan_directory_button = ttk.Button(self.frame, text="Select Directory to Scan", command=self.select_directory)
        self.scan_directory_button.grid(row=1, column=0, padx=5, pady=5)
        
        self.quit_button = ttk.Button(self.frame, text="Quit", command=self.root.quit)
        self.quit_button.grid(row=2, column=0, padx=5, pady=5)

        # Label for displaying results
        self.result_label = ttk.Label(self.root, text="Results will be displayed here")
        self.result_label.grid(row=1, column=0, padx=10, pady=10)

    def select_files(self):
        files = filedialog.askopenfilenames(title="Select Files", filetypes=[("All Files", "*.*")])
        if files:
            self.process_files(files)
    
    def select_directory(self):
        directory = filedialog.askdirectory(title="Select Directory")
        if directory:
            self.process_directory(directory)

    def process_files(self, files):
        scan_times = []
        scan_results = []
        
        for file in files:
            if file:
                result, scan_time = self.scan_file(file)
                scan_times.append(scan_time)
                scan_results.append(result)
        
        self.plot_scan_times(scan_times)
        total_scan_time = sum(scan_times)
        result_text = "\n".join(scan_results)
        
        self.display_results(result_text, total_scan_time)

    def process_directory(self, directory):
        scan_results, total_scan_time = self.scan_directory(directory)
        self.display_results(scan_results, total_scan_time)

    def scan_file(self, file_path):
        start_time = time.time()
        result = subprocess.run(['C:\\Users\\Harish T\\hackathon\\antivirus\\antivirus_scanner.exe', file_path], capture_output=True, text=True)
        end_time = time.time()
        scan_time = end_time - start_time
        return result.stdout, scan_time

    def scan_directory(self, directory_path):
        scan_times = []
        scan_results = []

        total_files = sum(len(files) for _, _, files in os.walk(directory_path))
        scanned_files = 0

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                result, scan_time = self.scan_file(file_path)
                scan_times.append(scan_time)
                scan_results.append(result)

                scanned_files += 1
                progress = int((scanned_files / total_files) * 100)
                print(f"Scanning {file_path}: {progress}% completed")

        self.plot_scan_times(scan_times)
        total_scan_time = sum(scan_times)
        return "\n".join(scan_results), total_scan_time

    def plot_scan_times(self, scan_times):
        plt.figure()
        plt.plot(scan_times, marker='o')
        plt.title("Time Taken to Scan Files")
        plt.xlabel("File Index")
        plt.ylabel("Time (seconds)")
        plt.savefig('scan_times.png')
        plt.close()

    def display_results(self, result_text, total_time):
        # Display results in the label
        self.result_label.config(text=f"Results:\n{result_text}\n\nTotal Scan Time: {total_time} seconds")
        
        # Display scan times plot
        try:
            image = Image.open('scan_times.png')
            photo = ImageTk.PhotoImage(image)
            self.result_label.image = photo
            self.result_label.config(image=photo)
        except Exception as e:
            print(f"Error loading image: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ScannerApp(root)
    root.mainloop()
