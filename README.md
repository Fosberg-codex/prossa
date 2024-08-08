# Prossa

Prossa is an open-source library for checking data preprocessing techniques applicable to a dataset.


## Installation

You can install Prossa using pip:

```
pip install prossa

<!-- or in jupyter notebooks -->

!pip install prossa
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
```
#### Documentation.
To enhance your data preprocessing flow with Prossa, please refer to the documentation: [Docs](https://prossa.pages.dev/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

You can find the project repository on GitHub:
[GitHub Repository](https://github.com/Fosberg-codex/prossa)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
