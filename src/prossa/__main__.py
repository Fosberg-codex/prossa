import sys
import pandas as pd
from .analyzer import analyze_dataset

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m prossa <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    try:
        dataset = pd.read_csv(csv_path)
        analyze_dataset(dataset)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()