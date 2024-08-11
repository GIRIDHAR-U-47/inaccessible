import os
import subprocess
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from flask import Flask, request, render_template, jsonify, redirect, url_for
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class MalwareNet(nn.Module):
    def _init_(self):
        super(MalwareNet, self)._init_()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    df['appeared'] = pd.to_datetime(df['appeared']).astype('int64') // 10**9
    df['sha256'] = df['sha256'].apply(lambda x: int(x, 16) % 10**8)
    df = df[df['label'] >= 0]

    numeric_features = df[['sha256', 'appeared']].values
    labels = df['label'].values

    scaler = StandardScaler()
    features = scaler.fit_transform(numeric_features)

    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return features, labels


def train_model(dataset_path):
    model = MalwareNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data, labels = load_dataset(dataset_path)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "models\\trained_model.pt")
    print("Model training completed and saved.")


def plot_scan_times(scan_times):
    plt.figure()
    plt.plot(scan_times, marker='o')
    plt.title("Time Taken to Scan Files")
    plt.xlabel("File Index")
    plt.ylabel("Time (seconds)")
    plt.savefig('static/scan_times.png')


def scan_directory(directory_path):
    scan_times = []
    scan_results = []

    total_files = sum(len(files) for _, _, files in os.walk(directory_path))
    scanned_files = 0

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            start_time = time.time()
            result = subprocess.run(['C:\\Users\\Harish T\\hackathon\\antivirus\\antivirus_scanner.exe', file_path], capture_output=True, text=True)
            end_time = time.time()

            scan_time = end_time - start_time
            scan_times.append(scan_time)
            scan_results.append(result.stdout)

            scanned_files += 1
            progress = int((scanned_files / total_files) * 100)
            print(f"Scanning {file_path}: {progress}% completed")  # This will be replaced with actual progress updates on the frontend

    plot_scan_times(scan_times)
    total_scan_time = sum(scan_times)
    return "\n".join(scan_results), total_scan_time


def list_drives():
    drives = [f"{d}:\\" for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{d}:\\")]
    return drives


app = Flask(_name_)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files.getlist('file')
    scan_times = []

    for file in files:
        if file.filename == '':
            return jsonify({"error": "No selected file"})

        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        start_time = time.time()
        result = subprocess.run(['C:\\Users\\Harish T\\hackathon\\antivirus\\antivirus_scanner.exe', file_path], capture_output=True, text=True)
        end_time = time.time()

        scan_time = end_time - start_time
        scan_times.append(scan_time)

    plot_scan_times(scan_times)
    total_scan_time = sum(scan_times)

    return render_template('result.html', result=result.stdout, total_time=total_scan_time)


@app.route('/scan_device', methods=['GET', 'POST'])
def scan_device():
    if request.method == 'POST':
        scan_type = request.form.get('scan_type')
        if scan_type == 'whole_scan':
            scan_results, total_scan_time = scan_directory('C:\\')
            return render_template('result.html', result=scan_results, total_time=total_scan_time)
        elif scan_type == 'disk_select':
            selected_drive = request.form.get('drive')
            if selected_drive:
                scan_results, total_scan_time = scan_directory(selected_drive)
                return render_template('result.html', result=scan_results, total_time=total_scan_time)
            else:
                return render_template('disk_select.html', drives=list_drives())

    return render_template('scan_options.html')


@app.route('/select_disk', methods=['GET', 'POST'])
def select_disk():
    if request.method == 'POST':
        selected_drive = request.form.get('drive')
        return redirect(url_for('scan_device', drive=selected_drive))

    drives = list_drives()
    return render_template('disk_select.html', drives=drives)


if _name_ == "_main_":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)