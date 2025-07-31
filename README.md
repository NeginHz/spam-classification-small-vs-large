# Spam Detector Benchmark: Small vs Large Data

This repository contains a complete spam detection project using traditional machine learning methods. It explores how model performance changes when trained on small data versus large data. The project compares four models across multiple datasets and identifies the best model for real-world deployment.

---

## Project Structure

### Notebooks
- `01_Path1_SmallData_Training_and_Evaluation.ipynb`:  
  Trains Naive Bayes and SVM on a small SMS dataset (Dataset 1). Evaluates on validation/test sets and two external datasets.
  
- `02_Path2_LargeData_Training_and_Evaluation.ipynb`:  
  Trains the same models on a larger dataset (Dataset 2). Evaluates on unseen small and general datasets.
  
- `03_Combined_Evaluation_and_Final_Conclusion.ipynb`:  
  Compares all four models (NB-1, SVM-1, NB-2, SVM-2) across datasets and selects the best for deployment.

### Models and Artifacts
- `nb1_model.pkl`, `svm1_model.pkl`: Trained models from Path 1 (small dataset)  
- `nb2_model.pkl`, `svm2_model.pkl`: Trained models from Path 2 (large dataset)  
- `tfidf_vectorizer_path1.pkl`, `tfidf_vectorizer_path2.pkl`: TF-IDF vectorizers used in training  
- `df_general_clean.pkl`: Cleaned general dataset (Dataset 3) for final evaluation  
- `results_path1.csv`, `results_path2.csv`: Evaluation metrics across datasets  

---

## Dataset Information

### Dataset 1: Small Dataset (GitHub - justmarkham)  
A clean benchmark dataset with 5,574 labeled SMS messages (spam or ham).  
**Used for:** Training and validation in **Path 1**.  
**Source:** [justmarkham – GitHub Dataset](https://github.com/justmarkham/DAT8/blob/master/data/sms.tsv)

### Dataset 2: SMS Spam Collection (Kaggle)  
A large dataset (~10,000 messages) combining multiple SMS spam datasets.  
**Used for:** Training in **Path 2**, evaluation in **Path 1**.  
**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/tinu10kumar/sms-spam-dataset)

### Dataset 3: External Generalization Dataset (Mendeley)  
Independently collected messages used in academic research.  
**Used for:** Generalization testing in **both paths**.  
**Source:** [Mendeley Dataset](https://data.mendeley.com/datasets/f45bkkt8pr/1)

---

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Download NLTK Resources (first time only)

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

```

### Run Notebooks in Order

1. `01_Path1_SmallData_Training_and_Evaluation.ipynb`  
2. `02_Path2_LargeData_Training_and_Evaluation.ipynb`  
3. `03_Combined_Evaluation_and_Final_Conclusion.ipynb`  

---

## Models Compared

| Model  | Path   | Trained On     |
|--------|--------|----------------|
| NB-1   | Path 1 | Small dataset  |
| SVM-1  | Path 1 | Small dataset  |
| NB-2   | Path 2 | Large dataset  |
| SVM-2  | Path 2 | Large dataset  |

---

## Final Recommendation

**SVM-2**, trained on the large dataset, consistently achieved the highest F1 and recall scores on general/unseen datasets, making it the best choice for deployment.


### Credits

Project by: Negin Hezarjaribi

NLP Projct - IU INTERNATIONAL UNIVERSITY OF APPLIED SCIENCES (IU)
