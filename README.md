# Machine Learning Classifiers: KNN and SVC for シ & ツ Characters
![Static Badge](https://img.shields.io/badge/Python-3.13.17-brightgreen)
![GitHub repo size](https://img.shields.io/github/repo-size/VXGN/katakana-ocr-clasifier)
![GitHub repo file or directory count](https://img.shields.io/github/directory-file-count/VXGN/katakana-ocr-clasifier)

This project implements machine learning classifiers (KNN and SVC) to classify Japanese characters シ and ツ. The workflow includes loading and preprocessing image datasets, splitting data into training and testing sets, training classifiers, evaluating their performance, and visualizing results using confusion matrices.

Using dataset from [etlcdb](http://etlcdb.db.aist.go.jp/the-etl-character-database/), specifically ETL1C and ETL5C, which contains handwritten Katakana characters. So if you want to use this code, please download the dataset first. And extract the dataset to the `Datasets/` directory. Make sure the directory structure looks like this:
```
Datasets/
Datasets/shi
Datasets/tsu
```

## Installation

### **Requirements**
- Required libraries:
  - `numpy`
  - `opencv-python`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `pandas`

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/VXGN/katakana-ocr-clasifier.git
   cd your-repo-name
    ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and extract the dataset into the `Datasets/` directory.
4. Run the script:
   ```bash
   python main.py
   ```
## Todo
- [x] Method to classify シ and ツ characters
   - [x] Implement KNN classifier
   - [x] Implement SVC classifier
   - [ ] Implement Random Forest classifier
   - [ ] Implement Naive Bayes classifier
- [ ] Add more datasets
- [ ] Integrate with a web application 
