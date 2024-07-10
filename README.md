# Prossa

Prossa is a python library for checking data processing techniques applicable on a dataset.

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
outlier = outliers = check_outliers(df)
print(outlier)
```

For more detailed usage instructions, please refer to the documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
