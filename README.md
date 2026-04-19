# ✋ Hand Gesture Controlled Puzzle Game

An interactive computer vision-based puzzle game where users can create and solve puzzles using only hand gestures — no mouse or keyboard required.

---

## 🚀 Features

* Real-time hand tracking using **MediaPipe**
* Create a selection frame using both hands
* Pinch gesture to capture image from webcam
* Convert captured image into a puzzle
* Shuffle and solve the puzzle using gestures
* Smooth UI overlays with real-time feedback

---

## 🛠 Tech Stack

* **Python**
* **OpenCV**
* **MediaPipe**
* **NumPy**

---

## 🎮 How It Works

1. Use both hands to create a rectangular frame
2. Perform a **pinch gesture** to capture the selected area
3. The captured image is divided into puzzle tiles
4. Tiles are shuffled automatically
5. Solve the puzzle using hand gestures

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/gesture-puzzle-game.git
cd gesture-puzzle-game
pip install -r requirements.txt
python main.py
```

---

## 🧠 Gestures Used

* ✌️ Two hands → Create selection frame
* 🤏 Pinch → Capture image
* 👆 Point → Select tile
* 🤏 Pinch + move → Drag & place tile

---

## 📌 Future Improvements

* Multiple difficulty levels (3x3, 4x4, etc.)
* Sound effects & animations
* Gesture customization
* Multiplayer / leaderboard
* Mobile/Web version

---

## 💡 Inspiration

Built while exploring **computer vision** and gesture-based interaction to create a more natural and touchless user experience.

---

## 🙌 Contributing

Feel free to fork this repo and improve it! PRs are welcome.

---

## ⭐ Show Your Support

If you like this project, give it a ⭐ on GitHub!
