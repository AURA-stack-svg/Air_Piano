# 🎹 Air Piano – Touchless Music Meets Technology

A virtual piano played in mid-air using hand gestures. No physical keys, no sound complaints—just pure gesture-based musical interaction.

## 🚀 Overview
**Air Piano** reimagines how we interact with music by combining Computer Vision and audio synthesis. Using a standard webcam, the system tracks hand and finger movements in real-time, allowing users to play a virtual keyboard displayed on their screen.

## ✨ Key Features
- **Gesture Recognition**: Leverages **OpenCV** and **CVZone** for high-precision hand and finger tracking.
- **Dynamic Keyboard**: Displays a virtual piano with 14 white and 10 black keys, complete with real-time visual feedback.
- **Guided Learning**: Built-in song tutorials (Twinkle Twinkle, Mary Had a Little Lamb, etc.) with a scoring system for interactive learning.
- **Rich Audio Engine**: Generates realistic piano sounds in real-time with rich harmonics using **Pygame** and **NumPy**.
- **Auto-Calibration**: Automatically adjusts to your hand position and lighting conditions for a seamless experience.

---

## 🛠️ Tech Stack
- **Languages**: Python
- **Computer Vision**: OpenCV, CVZone (HandTrackingModule)
- **Audio Synthesis**: Pygame (Mixer), NumPy
- **Mathematics**: Vector tracking and velocity detection algorithms

---

## 📁 Project Structure
- `air_piano.py`: The core application script containing the tracking logic, UI rendering, and sound engine.

## ⚙️ Installation & Usage
1.  **Clone the Repo**:
    ```bash
    git clone https://github.com/AURA-stack-svg/Air_Piano.git
    ```
2.  **Install Dependencies**:
    ```bash
    pip install opencv-python cvzone pygame numpy
    ```
3.  **Run the App**:
    ```bash
    python air_piano.py
    ```
4.  **How to Play**:
    - Place your hand above the virtual keyboard to calibrate.
    - Move your fingers downward over a key to "press" it.
    - Use keys `1-3` to select songs and `SPACE` to toggle tutorial mode.

---

<div align="center">
  <i>Combining technology and creativity to reimagine musical interaction.</i>
</div>
