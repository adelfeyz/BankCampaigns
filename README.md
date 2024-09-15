# Bank Marketing Campaign - Predicting Term Deposit Subscription

This project aims to build machine learning models to predict whether a customer will subscribe to a term deposit based on various features related to their demographics, past behavior, and interactions during direct marketing campaigns. The data used in this project comes from the **Bank Marketing Dataset** available from the UCI Machine Learning Repository.

## Project Overview

The primary objective of this project is to develop a predictive model that can improve the efficiency of future marketing campaigns by identifying customers who are more likely to subscribe to a term deposit. This will help optimize resource allocation and reduce campaign costs while increasing success rates.

### Steps in the Project:
1. **Data Preprocessing**: Cleaning the dataset, handling missing values, and transforming categorical features using encoding techniques.
2. **Exploratory Data Analysis**: Understanding key trends and patterns in the data through visualization and correlation analysis.
3. **Feature Engineering**: Creating new features and adjusting existing ones to improve model performance.
4. **Model Training & Evaluation**:
   - Logistic Regression
   - K Nearest Neighbors (KNN)
   - Decision Trees
   - Support Vector Machine (SVM)
5. **Hyperparameter Tuning**: Using Grid Search to find the optimal hyperparameters for each model.
6. **Performance Evaluation**: Comparing models using metrics such as Accuracy, Precision, Recall, and F1 Score.

## Installation

To run this project, follow the steps below:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/bank-marketing-prediction.git
    cd bank-marketing-prediction
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the dataset from [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) and place it in the `data/` directory.

## Usage

1. Run the data preprocessing script:
    ```bash
    python preprocess_data.py
    ```

2. Train the models and evaluate performance:
    ```bash
    python train_models.py
    ```

3. Perform hyperparameter tuning with Grid Search:
    ```bash
    python hyperparameter_tuning.py
    ```

## Results

### Model Comparison:

| Model               | Train Time | Train Accuracy | Test Accuracy | Precision | Recall  | F1 Score |
|---------------------|------------|----------------|---------------|-----------|---------|----------|
| Logistic Regression  | 5.22 sec   | 89.65%         | 89.79%        | 0.6607    | 0.1670  | 0.2667   |
| KNN                 | 0.01 sec   | 90.74%         | 89.37%        | 0.6127    | 0.1964  | 0.2974   |
| Decision Tree       | 0.16 sec   | 98.23%         | 84.07%        | 0.6420    | 0.2348  | 0.3438   |
| SVM                 | 8.46 sec   | 89.66%         | 90.10%        | 0.6607    | 0.1670  | 0.2667   |

- **Best Model**: The **SVM** model achieved the highest test accuracy (90.10%) but took longer to train.
- **Precision vs Recall**: Precision is higher across models, but recall is lower, indicating the models are conservative in predicting the positive class.

## Hyperparameter Tuning

After performing grid search for hyperparameter tuning, the following results were found:
- **Best KNN Parameters**: `n_neighbors = 15` with a score of 89.52%.
- **Best Logistic Regression Parameters**: `C = 0.1` and `solver = 'liblinear'` with a score of 89.63%.

## Conclusion

This project demonstrates how machine learning models can be used to predict the success of marketing campaigns for term deposits. By tuning models and adjusting performance metrics, we were able to achieve high accuracy and better understand the trade-offs between precision and recall.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any improvements or suggestions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) for the dataset.
- Special thanks to [scikit-learn](https://scikit-learn.org/) for providing easy-to-use machine learning tools.
