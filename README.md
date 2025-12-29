AutoJudge: ML Problem Difficulty Predictor

AutoJudge is an intelligent dashboard that uses Machine Learning to analyze programming problems. It predicts whether a problem is Easy, Medium, or Hard and assigns a complexity score from 1–10.

How to Use
1. Clone the repository to your local machine.
2. Install dependencies: `pip install -r requirements.txt`
3. Train the models: Run `python src/train.py`. 
   Note: The `classifier.pkl` and `regressor.pkl` files are not included in this repository due to GitHub's 100MB file size limit. Running this script will generate them locally on your computer.
4. Launch the Dashboard: `streamlit run app.py`

Technology Stack
Language: Python
ML Library: Scikit-Learn (ExtraTrees Classifier & Regressor)
NLP: TF-IDF Vectorization
UI: Streamlit
