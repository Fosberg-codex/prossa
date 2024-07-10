import pandas as pd
import numpy as np
from scipy import stats

class DatasetAnalyzer:
    """
    A class for analyzing datasets and providing recommendations for data preprocessing.
    """

    def __init__(self, dataset):
        """
        Initialize the DatasetAnalyzer with a dataset.

        :param dataset: pandas DataFrame to be analyzed
        """
        self.dataset = dataset
        self.recommendations = {}

    def analyze(self):
        """
        Perform a comprehensive analysis of the dataset, checking various aspects and generating recommendations.
        """
        self.check_missing_values()
        self.check_outliers()
        self.check_data_types()
        self.check_scaling_encoding()
        self.check_categorical_data()
        self.check_constant_columns()
        self.check_imputation()

    def print_recommendations(self):
        """
        Print all recommendations generated during the analysis.
        """
        for category, items in self.recommendations.items():
            print(f"\n{category.upper()}:")
            for item in items:
                print(f"- {item}")

    def check_missing_values(self):
        """
        Check for missing values in the dataset and provide recommendations for handling them.
        """
        recommendations = []
        missing_values = self.dataset.isnull().sum()
        
        if missing_values.sum() > 0:
            recommendations.append("Dataset contains missing values.")
            for column, count in missing_values[missing_values > 0].items():
                percentage = (count / len(self.dataset)) * 100
                recommendations.append(f"Column '{column}' has {count} ({percentage:.2f}%) missing values.")
            
            recommendations.append("Consider the following techniques:")
            recommendations.append("- Remove rows with missing values using dropna()")
            recommendations.append("- Impute missing values using fillna() or more advanced techniques")
            recommendations.append("- Use algorithms that handle missing values (e.g., some tree-based models)")
        else:
            recommendations.append("No missing values found in the dataset.")
        
        self.recommendations['Missing Values'] = recommendations

    def check_outliers(self):
        """
        Check for outliers in numeric columns using the z-score method and provide recommendations.
        """
        recommendations = []
        numeric_columns = self.dataset.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            z_scores = np.abs(stats.zscore(self.dataset[column].dropna()))
            outliers = np.sum(z_scores > 3)
            if outliers > 0:
                recommendations.append(f"Column '{column}' has {outliers} potential outliers (using z-score > 3).")

        if recommendations:
            recommendations.append("Consider the following techniques:")
            recommendations.append("- Investigate and potentially remove outliers")
            recommendations.append("- Use robust scaling methods (e.g., RobustScaler)")
            recommendations.append("- Apply transformations (e.g., log transformation) to reduce the impact of outliers")
        else:
            recommendations.append("No significant outliers detected using the z-score method.")

        self.recommendations['Outliers'] = recommendations

    def check_data_types(self):
        """
        Check the data types of all columns and provide recommendations for appropriate type conversions.
        """
        recommendations = []
        dtypes = self.dataset.dtypes

        for column, dtype in dtypes.items():
            recommendations.append(f"Column '{column}' has data type: {dtype}")

        recommendations.append("Consider the following:")
        recommendations.append("- Ensure numeric columns are of the appropriate type (int, float)")
        recommendations.append("- Convert datetime columns to datetime type if not already")
        recommendations.append("- Check for any unexpected data types")

        self.recommendations['Data Types'] = recommendations

    def check_scaling_encoding(self):
        """
        Provide recommendations for scaling numeric features and encoding categorical features.
        """
        recommendations = []
        numeric_columns = self.dataset.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) > 0:
            recommendations.append("Consider scaling numeric features:")
            recommendations.append("- Use StandardScaler for normal distributions")
            recommendations.append("- Use MinMaxScaler to scale to a specific range")
            recommendations.append("- Use RobustScaler if outliers are present")

        categorical_columns = self.dataset.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            recommendations.append("Consider encoding categorical features:")
            recommendations.append("- Use OneHotEncoder for nominal categorical data")
            recommendations.append("- Use OrdinalEncoder for ordinal categorical data")

        self.recommendations['Scaling and Encoding'] = recommendations

    def check_categorical_data(self):
        """
        Analyze categorical columns and provide recommendations for handling high-cardinality features.
        """
        recommendations = []
        categorical_columns = self.dataset.select_dtypes(include=['object', 'category']).columns

        for column in categorical_columns:
            unique_values = self.dataset[column].nunique()
            recommendations.append(f"Column '{column}' has {unique_values} unique categories.")

            if unique_values > 10:
                recommendations.append(f"  - Consider grouping less frequent categories in '{column}'")
            
        if len(categorical_columns) > 0:
            recommendations.append("General recommendations for categorical data:")
            recommendations.append("- Use label encoding for ordinal categories")
            recommendations.append("- Use one-hot encoding for nominal categories with few unique values")
            recommendations.append("- Consider feature hashing for high-cardinality categorical data")

        self.recommendations['Categorical Data'] = recommendations

    def check_constant_columns(self):
        """
        Identify columns with constant values and recommend their removal.
        """
        recommendations = []
        constant_columns = [column for column in self.dataset.columns if self.dataset[column].nunique() <= 1]

        if constant_columns:
            recommendations.append("The following columns have constant values:")
            recommendations.extend(constant_columns)
            recommendations.append("Consider removing these columns as they don't provide any information.")
        else:
            recommendations.append("No constant columns found in the dataset.")

        self.recommendations['Constant Columns'] = recommendations

    def check_imputation(self):
        """
        Provide recommendations for imputing missing values if they exist in the dataset.
        """
        recommendations = []
        missing_values = self.dataset.isnull().sum()

        if missing_values.sum() > 0:
            recommendations.append(" The data has missing imputations, consider the following imputation techniques:")
            recommendations.append("- Simple imputation: mean, median, or mode")
            recommendations.append("- Advanced imputation: KNN imputer, MICE, or domain-specific methods")
            recommendations.append("- For time series data, consider forward fill or backward fill")
            recommendations.append("- Create a binary column to indicate where values were imputed")
        else:
            recommendations.append("No missing values found, imputation is not necessary.")

        self.recommendations['Imputation'] = recommendations


def analyze_dataset(dataset):
    """
    Analyze a dataset using the DatasetAnalyzer class and print the recommendations.

    :param dataset: pandas DataFrame to be analyzed
    :return: DatasetAnalyzer instance with completed analysis
    """
    analyzer = DatasetAnalyzer(dataset)
    analyzer.analyze()
    analyzer.print_recommendations()
    return analyzer

# The following functions are convenience wrappers for individual checks

def check_missing_values(dataset):
    """
    Check for missing values in the dataset.

    :param dataset: pandas DataFrame to be analyzed
    :return: recommendation for missing values
    """
    analyzer = DatasetAnalyzer(dataset)
    analyzer.check_missing_values()
    
    recommendations = analyzer.recommendations['Missing Values']
    print('\n'.join(recommendations))

def check_outliers(dataset):
    """
    Check for outliers in the dataset.

    :param dataset: pandas DataFrame to be analyzed
    :return: recommendation for outliers
    """
    analyzer = DatasetAnalyzer(dataset)
    analyzer.check_outliers()
    recommendations = analyzer.recommendations['Outliers']
    print('\n'.join(recommendations))

def check_data_types(dataset):
    """
    Check data types in the dataset.

    :param dataset: pandas DataFrame to be analyzed
    :return: recommendation for data types
    """
    analyzer = DatasetAnalyzer(dataset)
    analyzer.check_data_types()
    recommendations = analyzer.recommendations['Data Types']

    # Join the list items into a single string, separating them with newlines
    print('\n'.join(recommendations))
    

def check_scaling_encoding(dataset):
    """
    Check for scaling and encoding recommendations.

    :param dataset: pandas DataFrame to be analyzed
    :return: recommendation for scaling and encoding
    """
    analyzer = DatasetAnalyzer(dataset)
    analyzer.check_scaling_encoding()
    recommendations = analyzer.recommendations['Scaling and Encoding']
    print('\n'.join(recommendations))


def check_categorical_data(dataset):
    """
    Check categorical data in the dataset.

    :param dataset: pandas DataFrame to be analyzed
    :return: recommendation for categorical data
    """
    analyzer = DatasetAnalyzer(dataset)
    analyzer.check_categorical_data()
    recommendation = analyzer.recommendations['Categorical Data']
    print('\n'.join(recommendation))

def check_constant_columns(dataset):
    """
    Check for constant columns in the dataset.

    :param dataset: pandas DataFrame to be analyzed
    :return: recommendation for constant columns
    """
    analyzer = DatasetAnalyzer(dataset)
    analyzer.check_constant_columns()
    constant_columns = analyzer.recommendations['Constant Columns']
    
    if constant_columns:
        return f"The following columns have constant values: {', '.join(constant_columns)}"
    else:
        return "No constant columns found in the dataset."

def check_imputation(dataset):
    """
    Check for imputation recommendations.

    :param dataset: pandas DataFrame to be analyzed
    :return: recommendation for imputation
    """
    analyzer = DatasetAnalyzer(dataset)
    analyzer.check_imputation()
    recommendations = analyzer.recommendations['Imputation']
    print('\n'.join(recommendations))
    
