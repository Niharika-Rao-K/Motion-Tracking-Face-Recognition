# Motion-Tracking-Face-Recognition
This project implements an intelligent motion-activated pan–tilt tracking system using a Raspberry Pi, servo motors, OpenCV, and LBPH face recognition.
<h1>🛰️ Raspberry Pi Motion-Triggered Face Recognition & Auto-Tracking System</h1>
<h3>🎯 <i>Real-time Motion Detection + Servo-based Pan/Tilt Tracking + LBPH Face Recognition</i></h3>

<p>
This project implements a <b>complete real-time surveillance system</b> using 
<strong>Raspberry Pi</strong>, <strong>OpenCV</strong>, <strong>LBPH face recognition</strong>, 
and <strong>dual-servo pan/tilt tracking</strong>.  
The camera activates when motion is detected, tracks movement smoothly using servos, 
detects faces, recognizes known individuals, and saves logs of recognized/unrecognized faces.
</p>

<hr>

<h2>🚀 Features</h2>

<h3>🔹 Motion Detection</h3>
<ul>
  <li>Frame differencing + contour analysis</li>
  <li>Detects movement only when the subject is large enough</li>
  <li>Ignores noise, shadows, and small objects</li>
</ul>

<h3>🔹 Servo-based Auto-Tracking</h3>
<ul>
  <li>Pan & tilt servos follow moving objects smoothly</li>
  <li>Low-pass filtered to avoid jitter</li>
  <li>Auto-centers after inactivity</li>
</ul>

<h3>🔹 Face Recognition</h3>
<ul>
  <li>LBPH algorithm for fast on-device recognition</li>
  <li>Trains automatically from folders in <code>/dataset/</code></li>
  <li>Displays real names based on folder names</li>
  <li>Saves 3 snapshots per detection</li>
</ul>

<h3>🔹 Organized Dataset Structure</h3>
<pre>
dataset/
   ├── Alice/
   │      ├── 0.jpg
   │      ├── 1.jpg
   ├── Bob/
   │      ├── 0.jpg
   │      ├── 1.jpg
</pre>

<h3>🔹 Logging System</h3>
<pre>
logs/
   ├── recognized/
   │      ├── Alice_20250207-110355_1.jpg
   │      ├── Alice_20250207-110355_2.jpg
   ├── unrecognized/
          ├── Unknown_20250207-110510_1.jpg
</pre>

<hr>

<h2>🛠️ Hardware Requirements</h2>
<table>
  <tr><th>Component</th><th>Quantity</th></tr>
  <tr><td>Raspberry Pi 4 / 3B+</td><td>1</td></tr>
  <tr><td>Pi Camera Module</td><td>1</td></tr>
  <tr><td>SG90/MG995 Pan Servo</td><td>1</td></tr>
  <tr><td>SG90/MG995 Tilt Servo</td><td>1</td></tr>
  <tr><td>External 5V Servo Power</td><td>1</td></tr>
  <tr><td>Jumper Wires</td><td>10</td></tr>
</table>

<hr>

<h2>🔌 Circuit Diagram</h2>
<p><i><img width="669" height="603" alt="image" src="https://github.com/user-attachments/assets/b66a27cd-75ea-43f2-bf9b-bc6bc865870c"><i></p>

<hr>

<h2>📂 Project Structure</h2>
<pre>
📁 Face-Tracking-Recognition
│
├── capture_images.py        # Capture images + Train LBPH model
├── motion_test_now.py       # Motion detection + servo tracking + face recognition
├── haarcascade_frontalface_default.xml
├── face_model.yml           # Auto-generated LBPH model
├── dataset/                 # Training images
└── logs/                    # Saved recognized/unrecognized captures
</pre>

<hr>

<h2>📸 1. Capture Images & Train Model</h2>

<p>Run:</p>
<pre><code>python3 capture_images.py</code></pre>

<p>You will be asked for a name:</p>
<pre>Enter the person name: Alice</pre>

<p>The script will:</p>
<ul>
  <li>Capture 30 aligned grayscale images</li>
  <li>Create folder in <code>/dataset/&lt;name&gt;/</code></li>
  <li>Train LBPH model automatically</li>
  <li>Save model as <code>face_model.yml</code></li>
  <li>Display label mapping</li>
</ul>

<hr>

<h2>🎥 2. Run Motion + Face Recognition + Tracking</h2>

<p>Run:</p>
<pre><code>python3 motion_test_now.py</code></pre>

<p>The system will:</p>
<ul>
  <li>Detect motion</li>
  <li>Track object using servo motors</li>
  <li>Detect/recognize faces</li>
  <li>Save 3 log images for each detection</li>
  <li>Display live preview with overlays</li>
</ul>

<p>Press <b>q</b> to stop.</p>

<hr>

<h2>⚙️ Key Parameters You Can Adjust</h2>
<table>
  <tr><th>Parameter</th><th>Default</th><th>Description</th></tr>
  <tr><td>MOVE_DEADZONE</td><td>15</td><td>Ignore tiny movements</td></tr>
  <tr><td>MIN_MOTION_AREA</td><td>6000</td><td>Minimum bounding box to treat as motion</td></tr>
  <tr><td>MIN_FACE_AREA</td><td>1500</td><td>Ignore very small faces</td></tr>
  <tr><td>KP_FAST</td><td>0.02</td><td>Servo responsiveness</td></tr>
  <tr><td>RECOGNITION_CONFIDENCE</td><td>70</td><td>Lower = stricter recognition</td></tr>
</table>

<hr>

<h2>📌 Highlights of This System</h2>
<ul>
  <li>Smooth, jitter-free servo tracking</li>
  <li>Real-time performance on Raspberry Pi</li>
  <li>No cloud required &mdash; everything local</li>
  <li>Structured dataset + logging</li>
  <li>Auto-mapping names from folders</li>
  <li>Lightweight LBPH model</li>
</ul>

<hr>

