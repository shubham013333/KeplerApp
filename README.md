

---

# 🌌 Kepler Exoplanet App

An interactive web app built with **Streamlit** and **Machine Learning** to predict whether a candidate observed by NASA’s Kepler Space Telescope is a **real exoplanet** or a **false positive**.

---

## Demo

👉 **Live App:** [Streamlit Deployment Link](https://shubham013333-keplerapp-srcstreamlit-app-zfumng.streamlit.app/)
👉 **Demo Video:** [YouTube Demo](https://www.youtube.com/watch?v=0tShQuAkGTc) 

<img width="977" height="669" alt="image" src="https://github.com/user-attachments/assets/d196e055-47be-4fa6-9492-167dad25af4b" />


---

## 📖 Project Overview

* Uses **NASA Kepler Dataset** (publicly available).
* Trained a **Machine Learning Pipeline** (`model_piepline.pk`) with algorithms like Random Forest and XGBoost.
* Predicts **CONFIRMED**, **CANDIDATE**, or **FALSE POSITIVE** based on orbital and planetary parameters.
* Fully interactive via **Streamlit web interface**.

---

## ⚙️ Features

* Input planetary system parameters such as:

  * Orbital Period (`koi_period`)
  * Planet Radius (`koi_prad`)
  * Stellar Flux (`koi_insol`)
  * Equilibrium Temperature (`koi_teq`)
* Instant classification/prediction of **planetary status**.
* Visualizations of data distributions and model insights.
* Easy-to-use interface – no coding required.

---

## 🧠 Tech Stack

* **Python**
* **Scikit-learn / XGBoost**
* **Joblib (Model Pipeline)**
* **Pandas & NumPy**
* **Streamlit**

---

## 📦 Installation & Usage

```bash
# Clone the repo
git clone https://github.com/shubham013333/KeplerApp.git
cd keplerApp

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run src/streamlit_app.py
```

Then open **[http://localhost:8501](http://localhost:8501)** in your browser. 🚀

---

## 🌍 Why It Matters

This project helps scientists, students, and enthusiasts quickly evaluate potential exoplanets from NASA’s Kepler mission. By simplifying access to machine learning predictions, we make space discovery more engaging and accessible to all.

---

## 👨‍💻 Contributors

* [Shubham Shrivas](https://github.com/shubham013333)

---

✨ *Inspired by NASA’s mission to discover worlds beyond our solar system.*

---

