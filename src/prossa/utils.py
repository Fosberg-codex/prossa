from .analyzer import DatasetAnalyzer

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