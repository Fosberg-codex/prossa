# Prossa

Prossa is an open-source library for checking data preprocessing techniques applicable on a dataset.

## Installation

You can install Prossa using pip:

```
pip install prossa
```

## Usage

Here's a quick example of how to use Prossa:

```python
import pandas as pd
from prossa import analyze_dataset
from prossa import check_outliers

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Analyze the dataset
analyze_dataset(df)

#also you can check if there are outliers in the dataset
outliers = check_outliers(df)
print(outliers)
```
### Methods in current prossa version

```python
# All methods take dataframe as an argument. ie. arg = df or arg = dataset

#Perform a comprehensive analysis of the dataset, checking various techniques in data preprocessing for recommendations.
analyze_dataset(arg)

#Check for missing values in the dataset.
check_missing_values(arg)

#Check for outliers in the dataset.
check_outliers(arg)

#Check data types in the dataset.
check_data_types(arg)

#Check if dataset needs scaling and encoding.
check_scaling_encoding(arg)

#Check categorical data in the dataset.
check_categorical_data(arg)

#Check for constant columns in the dataset.
check_constant_columns(arg)

#Check for imputation and get recommendations.
check_imputation(arg)


```


For more detailed usage instructions, please refer to the documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

You can find the project repository on GitHub:
[GitHub Repository](https://github.com/Fosberg-codex/prossa)

## License

This project is licensed under the MIT License - see the LICENSE file for details.