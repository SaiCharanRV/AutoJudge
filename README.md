# ⚖️ AutoJudge: Problem Difficulty Predictor

AutoJudge is an intelligent machine learning dashboard designed to analyze programming problem descriptions. The system predicts the categorical difficulty level (Easy, Medium, Hard) and assigns a numerical complexity score (1–10) based on textual and algorithmic features.

## 🛠️ Technical Approach
The core engine evaluates problems using a **Hybrid Feature Matrix**:
* **NLP Vectorization**: Utilizes word-level **TF-IDF** to capture the significance of algorithmic phrasing and keywords.
* **Domain Feature Extraction**: Custom logic identifies specific markers like `recursion`, `dynamic programming`, and data structure frequencies.
* **Machine Learning Models**: Employs **ExtraTrees** (Extremely Randomized Trees) for both the classification and regression tasks to ensure robust and accurate predictions.

## 📂 Project Structure
The repository is organized as:
* **`app.py`**: The primary Streamlit dashboard for real-time user interaction.
* **`src/`**: Contains `train.py` for model training and `features.py` for feature engineering logic.
* **`data/`**: Includes the `raw/` dataset and `processed/` TF-IDF vectorizer.
* **`requirements.txt`**: Lists all necessary Python libraries to reproduce the environment.

## 💻 How to Run
1. **Clone the repository**
2. **Install dependencies**:  pip install -r requirements.txt
3. **Train the Models Locally**: The trained model files (classifier.pkl and regressor.pkl) are not included in this repository because they exceed GitHub's 100MB file size limit. Run the training script to generate them on your machine:  python src/train.py
4. **Launch the Dashboard**:   streamlit run app.py
   
  
