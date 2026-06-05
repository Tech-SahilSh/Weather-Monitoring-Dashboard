import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from datetime import datetime

# Data and model paths
data = pd.read_csv("weather-data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')

MODEL_FILES = {
    'Linear Regression': 'linear_model.pkl',
    'Logistic Regression': 'logistic_model.pkl',
    'Random Forest': 'random_forest_model.pkl',
    'Decision Tree': 'decision_tree_model.pkl',
}

# Helper to get X, y
def get_X_y():
    X = data[['Temp_Max', 'Temp_Min', 'Humidity', 'Heatwave']]
    y = data['RainFlag']
    return X, y

def evaluate_models():
    X, y = get_X_y()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scores = {}
    for name, file in MODEL_FILES.items():
        try:
            model = joblib.load(file)
            if name == 'Linear Regression':
                y_pred = model.predict(X_test)
                y_pred = np.round(y_pred)
                score = accuracy_score(y_test, y_pred)
            else:
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
            scores[name] = score
        except Exception as e:
            scores[name] = 0
    return scores

def get_best_model():
    scores = evaluate_models()
    best = max(scores, key=scores.get)
    return best, scores[best]

def plot_model(model_name):
    fig, ax = plt.subplots(figsize=(8, 3))
    color_map = {
        'Linear Regression': ('royalblue', 'skyblue'),
        'Logistic Regression': ('seagreen', 'lightgreen'),
        'Random Forest': ('darkorange', 'gold'),
        'Decision Tree': ('purple', 'violet'),
    }
    color, fill = color_map.get(model_name, ('gray', 'lightgray'))
    ax.plot(data['Date'], data['Rain_mm'], color=color)
    ax.fill_between(data['Date'], data['Rain_mm'], color=fill, alpha=0.4)
    ax.set_title(f"🌧️ Rainfall Over Time ({model_name})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rain_mm")
    ax.grid(True)
    fig.tight_layout()
    return fig

def open_predict_window(best_model_name):
    pred_win = tk.Toplevel()
    pred_win.title(f"Predict Rain (Best Model: {best_model_name})")
    pred_win.geometry("350x400")
    pred_win.configure(bg='black')
    fields = ['Temp_Max', 'Temp_Min', 'Humidity', 'Heatwave']
    entries = {}
    for i, field in enumerate(fields):
        tk.Label(pred_win, text=field+':', bg='black', fg='#fff', font=("Arial", 12)).grid(row=i, column=0, padx=10, pady=10, sticky='e')
        entry = tk.Entry(pred_win, font=("Arial", 12))
        entry.grid(row=i, column=1, padx=10, pady=10)
        entries[field] = entry
    def predict():
        try:
            values = [float(entries[f].get()) for f in fields]
            model = joblib.load(MODEL_FILES[best_model_name])
            arr = np.array(values).reshape(1, -1)
            pred = model.predict(arr)
            if best_model_name == 'Linear Regression':
                rain_prob = float(pred[0])
                rain_flag = int(round(rain_prob))
            else:
                rain_flag = int(pred[0])
            # Predict rain mm using a regression model if available
            try:
                rain_mm_model = joblib.load('linear_model.pkl')
                rain_mm = rain_mm_model.predict(arr)[0]
            except Exception:
                rain_mm = 'N/A'
            result = 'Rain' if rain_flag == 1 else 'No Rain'
            msg = f"Prediction: {result}\nPredicted Rainfall: {rain_mm:.2f} mm" if isinstance(rain_mm, float) else f"Prediction: {result}\nPredicted Rainfall: {rain_mm} mm"
            messagebox.showinfo("Prediction", msg)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    tk.Button(pred_win, text="Predict", command=predict, font=("Arial", 12, "bold"), bg='black', fg='#0d47a1', bd=0, highlightthickness=0, activebackground='black', activeforeground='#1976d2').grid(row=len(fields), column=0, columnspan=2, pady=20)

class CloudButton(tk.Canvas):
    def __init__(self, master, text, command=None, width=200, height=60, font=("Arial", 14, "bold"), **kwargs):
        super().__init__(master, width=width, height=height, bg='white', highlightthickness=0, **kwargs)
        self.command = command
        self.text = text
        self.font = font
        # Draw cloud shape
        self.create_oval(10, 20, 60, 60, fill='white', outline='white')
        self.create_oval(50, 10, 150, 60, fill='white', outline='white')
        self.create_oval(130, 20, 190, 60, fill='white', outline='white')
        self.create_rectangle(30, 35, 170, 60, fill='white', outline='white')
        self.text_id = self.create_text(width//2, height//2+5, text=text, font=font, fill='#0d47a1')
        self.bind('<Button-1>', self._on_click)
        self.tag_bind(self.text_id, '<Button-1>', self._on_click)
    def _on_click(self, event):
        if self.command:
            self.command()

class RainPredictorApp:
    def __init__(self, root):
        root.configure(bg='black')
        self.root = root
        root.title("Rain Predictor - Model Visualizer")
        root.geometry("600x500")
        # Title at the top
        title = tk.Label(root, text="Rain Predictor", font=("Arial", 22, "bold"), fg='#1565c0', bg='black')
        title.pack(pady=(20, 10))
        # Location and live temperature below the title
        self.location_label = tk.Label(root, text="Pune, India", font=("Arial", 14, "bold"), fg='#1565c0', bg='black')
        self.location_label.pack()
        self.temp_label = tk.Label(root, text="Temp: -- °C", font=("Arial", 14, "bold"), fg='#1565c0', bg='black')
        self.temp_label.pack(pady=(0, 5))
        self.update_temperature()
        # Live clock below temperature, matching color
        self.clock_label = tk.Label(root, font=("Arial", 18, "bold"), fg='#1565c0', bg='black')
        self.clock_label.pack(pady=(0, 20))
        self.update_clock()
        # All buttons as text-only (no background), stacked vertically with extra gap
        btn_frame = tk.Frame(root, highlightthickness=0, bd=0, bg='black')
        btn_frame.pack(pady=10)
        # Model visualization buttons (text only, no bg, no border, dark blue color, transparent)
        for model_name in MODEL_FILES.keys():
            btn = tk.Button(btn_frame, text=f"Show {model_name} Visualization", font=("Arial", 15, "bold"),
                            fg='#0d47a1', bd=0, highlightthickness=0, activebackground='black', activeforeground='#1976d2',
                            cursor='hand2', relief='flat', bg='black',
                            command=lambda m=model_name: self.open_visualization_window(m))
            btn.pack(pady=18)
        # Predict button (text only)
        pred_btn = tk.Button(btn_frame, text="Predict using Best Model", font=("Arial", 15, "bold"),
                            fg='#0d47a1', bd=0, highlightthickness=0, activebackground='black', activeforeground='#1976d2',
                            cursor='hand2', relief='flat', bg='black',
                            command=self.predict_best)
        pred_btn.pack(pady=22)

    def update_clock(self):
        import time
        now = time.strftime('%H:%M:%S')
        self.clock_label.config(text=now)
        self.clock_label.after(500, self.update_clock)

    def update_temperature(self):
        import requests
        try:
            # Use Open-Meteo API for current temperature in Pune
            url = "https://api.open-meteo.com/v1/forecast?latitude=18.52&longitude=73.86&current_weather=true"
            resp = requests.get(url, timeout=5)
            temp = '--'
            if resp.ok:
                data = resp.json()
                temp = data.get('current_weather', {}).get('temperature', '--')
            self.temp_label.config(text=f"Temp: {temp} °C")
        except Exception:
            self.temp_label.config(text="Temp: -- °C")
        self.temp_label.after(60000, self.update_temperature)

    def open_model_menu(self):
        menu_win = tk.Toplevel(self.root)
        menu_win.title("Model Visualizations")
        menu_win.geometry("400x350")
        menu_win.configure(bg='black')
        for i, model_name in enumerate(MODEL_FILES.keys()):
            btn = tk.Button(menu_win, text=f"Show {model_name}", width=30, font=("Arial", 13, "bold"),
                            fg='#0d47a1', bg='black', bd=0, highlightthickness=0, activebackground='black', activeforeground='#1976d2',
                            cursor='hand2', relief='flat',
                            command=lambda m=model_name: self.open_visualization_window(m))
            btn.pack(pady=10)
        back_btn = tk.Button(menu_win, text="Back", command=menu_win.destroy,
                             font=("Arial", 11), bg='black', fg='#01579b',
                             activebackground='black', activeforeground='#81d4fa', bd=0, highlightthickness=0)
        back_btn.pack(pady=10)

    def open_visualization_window(self, model_name):
        vis_win = tk.Toplevel(self.root)
        vis_win.title(f"{model_name} Visualization")
        vis_win.geometry("900x400")
        vis_win.configure(bg='black')
        fig = plot_model(model_name)
        canvas = FigureCanvasTkAgg(fig, master=vis_win)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10, fill='both', expand=True)
        back_btn = tk.Button(vis_win, text="Back", command=vis_win.destroy,
                             font=("Arial", 11), bg='black', fg='#01579b',
                             activebackground='black', activeforeground='#81d4fa', bd=0, highlightthickness=0)
        back_btn.pack(pady=10)

    def predict_best(self):
        best_model, score = get_best_model()
        open_predict_window(best_model)

if __name__ == "__main__":
    root = tk.Tk()
    app = RainPredictorApp(root)
    root.mainloop()
