# Predicting Future Store Sales with AI 📈

This project tackles the challenge of forecasting daily sales for Rossmann stores using historical sales data, store information, and promotional data. It employs time series analysis techniques and the powerful XGBoost algorithm to build a predictive regression model.

This notebook demonstrates an end-to-end forecasting workflow, covering data merging and cleaning, extensive feature engineering tailored for time series data, exploratory data analysis (EDA), model training, evaluation using a competition-specific metric (RMSPE), and generating predictions for submission.

**Dataset:** Rossmann Store Sales (includes `train.csv`, `test.csv`, `store.csv` - likely from the Kaggle competition)
**Focus:** Demonstrating time series feature engineering, regression modeling with XGBoost for forecasting, handling missing values, and evaluating predictions using RMSPE.
**Repository:** [https://github.com/Jayasurya227/Predicting-Future-Store-Sales-with-AI](https://github.com/Jayasurya227/Predicting-Future-Store-Sales-with-AI)

***

## Key Techniques & Concepts Demonstrated

Based on the analysis within the notebook (`6_Predicting_Future_Store_Sales_with_AI.ipynb`), the following key concepts and techniques are applied:

* **Time Series Forecasting:** Building a model to predict future values (store sales) based on historical data.
* **Data Merging & Cleaning:**
    * Combining `train.csv`/`test.csv` with `store.csv` to enrich sales data with store-specific attributes.
    * Converting date columns to datetime objects.
    * Handling missing values, notably imputing `CompetitionDistance` with the median.
* **Exploratory Data Analysis (EDA) for Time Series:**
    * Visualizing sales trends over time.
    * Analyzing the impact of promotions, holidays, and store types on sales using various plots (boxplots, countplots).
    * Examining correlations between numerical features.
* **Advanced Feature Engineering for Time Series:**
    * Extracting multiple date-based features: `Year`, `Month`, `Day`, `WeekOfYear`, `DayOfWeek`.
    * Creating features related to competition: `CompetitionOpenSinceMonth`, `CompetitionOpenSinceYear`, `CompetitionOpen`.
    * Creating features related to promotions: `PromoOpen`, `PromoInterval` mapping.
    * Encoding categorical features: Manually mapping `StoreType`, `Assortment`, and `StateHoliday` to numerical representations.
* **Target Transformation:** Applying a **log transformation** (`np.log1p`) to the `Sales` target variable to handle potential skewness and stabilize variance, common in sales data.
* **Model Building (XGBoost):**
    * Selecting relevant engineered features.
    * Splitting the training data into training and validation sets based on date to simulate a real forecasting scenario (predicting a future period).
    * Training an `XGBRegressor` model.
* **Custom Evaluation Metric (RMSPE):** Implementing and using the Root Mean Squared Percentage Error, the evaluation metric specific to the Rossmann Sales Kaggle competition, to assess model performance on the validation set (after reversing the log transform using `np.expm1`).
* **Prediction & Submission:** Training the final XGBoost model on the entire relevant training dataset, making predictions on the test set, reversing the log transformation, and formatting the output into a `submission.csv` file.

***

## Analysis Workflow

The notebook follows a structured forecasting workflow:

1.  **Setup & Data Loading:** Importing libraries (Pandas, NumPy, Matplotlib, Seaborn, XGBoost, Scikit-learn) and loading the three dataset files (`train.csv`, `test.csv`, `store.csv`).
2.  **Data Cleaning & Merging:**
    * Converting 'Date' columns to datetime objects.
    * Merging store information into the train and test datasets based on `Store` ID.
    * Handling missing values (e.g., median imputation for `CompetitionDistance`).
3.  **Exploratory Data Analysis (EDA):**
    * Visualizing overall sales trends.
    * Analyzing sales patterns based on `StoreType`, `Assortment`, `Promo`, `DayOfWeek`, `StateHoliday`, etc.
    * Plotting correlations between numerical features.
4.  **Feature Engineering:**
    * Extracting various components from the 'Date' column.
    * Creating features related to competition opening times and promotion intervals.
    * Manually encoding categorical features (`StoreType`, `Assortment`, `StateHoliday`).
5.  **Model Preparation:**
    * Log-transforming the `Sales` target variable (`np.log1p`).
    * Selecting the final set of features for modeling.
    * Splitting the data chronologically into training and validation sets.
6.  **Model Training (XGBoost):**
    * Initializing and training an `XGBRegressor` model on the training set.
7.  **Model Evaluation:**
    * Making predictions on the validation set.
    * Reversing the log transformation (`np.expm1`) on both actual and predicted validation sales.
    * Calculating the Root Mean Squared Percentage Error (RMSPE) using a custom function.
8.  **Final Prediction & Submission:**
    * Training the XGBoost model on all available training data.
    * Making predictions on the processed test set.
    * Reversing the log transformation on the final predictions.
    * Creating the `submission.csv` file in the required format.

***

## Technologies Used

* **Python**
* **Pandas & NumPy:** For data loading, manipulation, cleaning, and feature engineering.
* **Matplotlib & Seaborn:** For data visualization.
* **Scikit-learn:** Primarily for data splitting (`train_test_split`, though used carefully for time series).
* **XGBoost:** For building the gradient boosting regression model (`XGBRegressor`).
* **Jupyter Notebook / Google Colab:** For the interactive analysis environment.

***

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Jayasurya227/Predicting-Future-Store-Sales-with-AI.git](https://github.com/Jayasurya227/Predicting-Future-Store-Sales-with-AI.git)
    cd Predicting-Future-Store-Sales-with-AI
    ```
2.  **Install dependencies:**
    (It is recommended to use a virtual environment)
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
    ```
3.  **Ensure Datasets:** Make sure the `train.csv`, `test.csv`, and `store.csv` files (from the Rossmann Store Sales competition) are present in the repository directory.
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook "6_Predicting_Future_Store_Sales_with_AI.ipynb"
    ```
    *(Run the cells sequentially to perform the analysis.)*

***

## Author & Portfolio Use

* **Author:** Jayasurya227
* **Portfolio:** This project ([https://github.com/Jayasurya227/Predicting-Future-Store-Sales-with-AI](https://github.com/Jayasurya227/Predicting-Future-Store-Sales-with-AI)) demonstrates practical skills in time series forecasting, feature engineering for temporal data, and applying advanced regression models like XGBoost. It's suitable for showcasing on GitHub, resumes/CVs, LinkedIn, and during interviews for data science or analytics roles involving forecasting.
* **Notes:** Recruiters can review the detailed EDA, the thoughtful creation of time-based features, the handling of missing data, the application of XGBoost, and the use of the competition-specific RMSPE metric for evaluation. The project addresses a common business problem of sales forecasting.
