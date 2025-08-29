# 🍎 Fruit Ripeness Classifier
A **Streamlit app** that predicts the ripeness of fruits using a **TensorFlow / Keras model**.  
The app supports `.h5` or `.tflite` models and allows users to upload fruit images and get instant ripeness predictions with confidence scores.

## 🚀 Live Demo  

The app is deployed temporarily using **ngrok**.  
**Ask me for the current live link!**

## 🖥 Features

- Upload fruit images (JPG, JPEG, PNG)  
- Predict ripeness: **Unripe, Ripe, Overripe**  
- Shows prediction confidence (%)  
- Lightweight inference with TensorFlow Lite  
- Optional public sharing using **ngrok**

## 📸 Screenshots
![App Screenshot](screenshots/app_screenshot.png)  
*Upload an image and get predictions instantly.*

## ⚙ Installation / Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "Fruit Ripeness Classifier"
````

2. Create a virtual environment:

```bash
python -m venv tfenv
```

3. Activate the environment:

**Windows (PowerShell):**

```powershell
.\tfenv\Scripts\activate
```

**Linux / Mac:**

```bash
source tfenv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the Streamlit app:

```bash
streamlit run Streamlitapp.py
```

6. Optional: Start **ngrok** for public sharing:

```python
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print("Public URL:", public_url)
```

---

## 🗂 Folder Structure

Fruit Ripeness Classifier/
├─ Models/
│  ├─ best_fruit_model.h5
│  └─ fruit_ripeness_model_final.h5
├─ Streamlitapp.py
├─ requirements.txt
├─ README.md
├─ screenshots/
└─ ...


## 🧠 Model Info

* Models: `.h5` or `.tflite`
* Input: RGB image, **150x150 pixels**
* Classes: `Unripe`, `Ripe`, `Overripe`
* Normalized pixel values (0-1)
* Trained using a custom fruit ripeness dataset

## 📝 Usage

1. Run the app with Streamlit
2. Upload a fruit image
3. Click **Predict Ripeness**
4. View predicted class and confidence score

## 🛠 Tech Stack

* Python 3.x
* TensorFlow / Keras
* TensorFlow Lite
* Streamlit
* pyngrok (for public access)
* Pillow & NumPy

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.


## 🙏 Acknowledgements

* TensorFlow and Keras tutorials for image classification
* Streamlit documentation for interactive UI
* ngrok documentation for public access




