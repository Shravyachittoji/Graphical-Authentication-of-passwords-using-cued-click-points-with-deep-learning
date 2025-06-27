

# Graphical Authenticaton of Passwords with Deep Learning

This project implements a secure graphical password authentication system that leverages deep learning models for enhanced security and user experience. The system allows users to select graphical passwords and evaluates the performance of different models using key metrics.

## Features

- Graphical password selection using click points on images
- Deep learning models (ResNet, ViT, Hybrid) for authentication
- Secure storage of user credentials
- Performance evaluation and visualization
- **Only two graphs are generated:**
  - `training_metrics.png` (IRR, PDS, and their standard deviations)
  - `precision_recall_f1_comparison.png` (Precision, Recall, F1-Score comparison)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Graphical-Authentication-of-passwords-using-cued-click-points-with-deep-learning
   cd graphical-password
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add authentication images:**
   - Place your images in the `images/` directory.

## Usage

1. **Run the application:**
   ```bash
   python main.py
   ```

2. **Register and login:**
   - Register a new user by entering a username and selecting click points on images.
   - Login using your username and click points.

3. **Generate performance graphs:**
   - The following graphs will be generated automatically after training:
     - `training_metrics.png`
     - `precision_recall_f1_comparison.png`

## Output Graphs

- **training_metrics.png:**  
  Shows Information Retention Rate (IRR), Password Diversity Score (PDS), and their standard deviations for all models in a 2x2 grid.

- **precision_recall_f1_comparison.png:**  
  Bar chart comparing Precision, Recall, and F1-Score for Hybrid, ResNet, ViT, and Traditional models.

## Requirements

- Python 3.7+
- PyQt5
- PyTorch
- OpenCV
- NumPy
- scikit-learn
- matplotlib

