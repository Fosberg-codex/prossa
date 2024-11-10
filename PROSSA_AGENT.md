# **Prossa Library Overview**

**Prossa** is a data preprocessing library designed to simplify and automate the preparation of datasets for machine learning workflows, with a focus on structured, tabular data (e.g., CSVs). Currently, Prossa operates primarily with manual commands, allowing users to preprocess data efficiently through a suite of well-defined tools and classes. The ultimate goal of Prossa is to enable data scientists to perform all necessary preprocessing steps in one environment, saving time and ensuring consistency across datasets.

#### **Current Manual Features**

1. **DataAnalyzer**: Provides a comprehensive analysis of datasets, including statistical summaries and data quality assessments, enabling users to understand dataset structure and detect initial data issues.

2. **OutlierDetector**: Identifies and handles outliers within a dataset using various detection methods, helping users clean up extreme or erroneous data points.

3. **MissingValueHandler**: Manages missing data by offering both imputation techniques and removal strategies, ensuring data completeness for reliable model training.

4. **Scaler**: Standardizes or normalizes numerical data to make it suitable for machine learning algorithms, particularly beneficial for algorithms sensitive to data scales.

5. **Encoder**: Transforms categorical variables into numerical formats, allowing them to be used effectively in machine learning models.

6. **FeatureSelector**: Selects relevant features based on statistical tests or model-based approaches, aiding in dimensionality reduction and improving model performance.

7. **DataVisualizer**: Generates visual representations of data distributions and relationships, allowing users to intuitively understand and communicate dataset characteristics.

8. **PipelineBuilder**: Constructs preprocessing pipelines to streamline data transformation workflows, enabling users to apply consistent steps across datasets.

These manual features empower Prossa users to handle data preprocessing efficiently, ensuring that datasets are ready for analysis and model training.

---

### **Your Role in Prossa’s Development**

As a **contributor to Prossa**, your role centers on evolving the library towards increased automation and efficiency. You’re responsible for enhancing the current capabilities of Prossa by integrating an **LLM-driven AI agent** to automate the identification, recommendation, and execution of preprocessing tasks. This involves creating an agent capable of independently analyzing datasets, generating preprocessing recommendations, and validating these recommendations before presenting them to the user.

### **Your Implementation Responsibilities**

You are in charge of implementing the following components within Prossa:

1. **Agent() Function Initialization**: Designing the Agent function to automatically select the most suitable LLM based on dataset complexity and availability, with dynamic prompts tailored to each preprocessing task.
  
2. **LLM Task Assignment**: Assigning roles to each LLM for specific preprocessing tasks, ensuring optimized use of each model’s capabilities.

3. **LLM Analysis and Recommendation Generation**: Enabling the agent to evaluate datasets and generate a set of preprocessing actions.

4. **Embedding Storage and Retrieval**: Storing LLM-generated recommendations as embeddings in a vector database, allowing the agent to "learn" over time for more efficient processing of similar datasets.

5. **Custom Validation (_confidence_validation)**: Developing and implementing a custom validation algorithm to verify the accuracy of the LLM’s recommendations, ensuring results meet a defined quality threshold before being returned to the user.

6. **User Report Generation**: Designing a verbose, detailed report for users, explaining each recommended preprocessing step and its anticipated impact on model readiness.

With these responsibilities, you’re positioning Prossa to transition from a manual preprocessing tool to a smart, autonomous agent that can handle preprocessing tasks with minimal user input.

---

### **Initial Implementation Breakdown**

#### **Agent() Function Initialization**
- **Overview**: When we call `Agent()`, it will auto-select the most suitable LLM based on dataset type, complexity, or LLM availability. This ensures resilience if certain models are temporarily inaccessible. The function will create an initial system prompt tailored to describe the dataset’s nature and potential preprocessing needs.
  
**Example**:
```python
from prossa_agent import Agent
```

#### **Available LLMs**:
- **Google Gemini API**:
  - *Gemini 1.5 Pro*: `gemini-1.5-pro`
  - *Gemini 1.5 Flash*: `gemini-1.5-flash`

- **OpenAI API**:
  - *GPT-4o*: `gpt-4o`
  - *o1-preview*: `o1-preview`

- **Anthropic API**:
  - *Claude 3.5 Sonnet*: `claude-3-5-sonnet-20241022`
  - *Claude 3.5 Haiku*: `claude-3.5-haiku`
  - *Claude 3.5 Sonnet -latest alias*: `claude-3-5-sonnet-latest` (for development and testing). For production, Anthropic recommends using a specific model version (e.g., `claude-3-5-sonnet-20241022`) to ensure consistent behavior.

---

### **Defining LLM Task Responsibilities**

Each LLM will be assigned specific tasks according to its strengths to ensure consistent performance.

- **Claude 3.5 Sonnet**:
  - **Role**: Advanced Data Analysis and Validation
  - **Tasks**: Handles high-stakes tasks like validating preprocessing steps, advanced statistical analysis, and consistency checks. Ideal for `_confidence_validation` due to its strong interpretative capabilities.
  - **Backup**: GPT-4o if unavailable.

- **Claude 3.5 Haiku**:
  - **Role**: Feature Engineering and Data Imbalance Solutions
  - **Tasks**: Creates feature engineering insights, addresses class imbalance issues (e.g., recommending resampling techniques), and interprets categorical data.
  - **Backup**: Claude 3.5 Sonnet if unavailable.

- **GPT-4o**:
  - **Role**: Core Data Analysis and Statistical Summaries
  - **Tasks**: Responsible for primary data analysis, outlier detection, and generating detailed statistical summaries. Our primary model for in-depth dataset analysis and initial recommendations.
  - **Backup**: Claude 3.5 Sonnet or Gemini 1.5 Pro if necessary.

- **o1-preview**:
  - **Role**: Scalable and General Preprocessing Recommendations
  - **Tasks**: Manages straightforward tasks such as missing value handling, normalization, and encoding. Acts as the versatile all-rounder for lower-complexity datasets.
  - **Backup**: GPT-4o or Gemini 1.5 Flash.

- **Gemini 1.5 Pro**:
  - **Role**: Lightweight Initial Analysis and Basic Recommendations
  - **Tasks**: Conducts basic dataset assessments and flags common preprocessing needs, such as missing values and categorical data, providing initial insights before in-depth analysis.
  - **Backup**: Gemini 1.5 Flash if unavailable.

- **Gemini 1.5 Flash**:
  - **Role**: Initial Dataset Screening and Cost-Saving Analysis
  - **Tasks**: Performs quick, cost-efficient assessments, identifying basic requirements (e.g., normalization needs, flagging obvious outliers). Primarily used for large datasets where an initial overview can help save costs before deeper analysis.
  - **Backup**: Any available LLM, prioritizing o1-preview or Claude 3.5 Haiku for initial assessments.

---

### **LLM Analysis and Recommendations**

- **Data Processing**: The agent passes the system prompt and dataset to the LLM, which, based on Prossa’s capabilities, generates recommended preprocessing actions.
- **Embedding Storage**: After analysis, recommendations are stored as embeddings in a vector database, tagged with dataset characteristics. This enables quick retrieval of context and previous results for similar datasets, allowing the agent to “learn” over time.

---

### **LLM Processing and Embedding Storage**

- **Vector Database Integration**: Embeddings are stored with detailed tags for dataset type and analysis step (e.g., `DataAnalyzer_Embedding`, `Scaler_Embedding`), allowing the Agent to quickly recall and reuse task-specific insights.
- **Embedding Granularity**: By storing embeddings by analysis step, the Agent can efficiently process similar datasets in the future, optimizing both time and resource use.

---

### **_confidence_validation for Quality Assurance**

- **Adaptive Thresholding**: The validation threshold, set initially at 0.9, can dynamically adjust based on model and task complexity. Certain tasks, like outlier detection, may require stricter validation.
- **Validation Logic**: `_confidence_validation` evaluates alignment with Prossa’s modules and prompt objectives. If validation falls below the threshold, it triggers a secondary analysis or prompts the user for additional input.

---

### **Returning Results to the User**

- **Report Delivery**: Upon validation, a detailed report of recommended actions is sent to the user, outlining each suggested step. This report does not yet apply changes to the dataset (e.g., filling missing values or applying feature engineering) but provides actionable insights.

### **Results Delivery and Reporting**

- **Verbose User Report**: The report explains each recommendation, enabling the user to understand the agent’s logic. Each recommendation includes:
  - **Action Justification**: A rationale for each step (e.g., why scaling or outlier removal is recommended).
  - **Impact Assessment**: Anticipated benefits for model readiness, allowing the user to make informed decisions about each action.
