# BULL FLAG DETECTOR PROJECT
As an Applied Mathematics student, my experience has focused on data science projects in Jupyter notebooks. This is my first project at this scale and my initial venture into Docker. **The models were trained on CPU, so they are intentionally designed for efficiency rather than maximum complexity.**

## Project Details

### Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: Kránitz Bence
- **Aiming for +1 Mark**: No

### Solution Description

This project addresses the challenge of automating technical analysis in financial markets, specifically the detection of complex chart patterns such as Bullish and Bearish Flags, Pennants, and Wedges. Manual identification of these patterns is subjective, time-consuming, and difficult to scale across multiple assets like EURUSD and Gold (XAU).

To solve this, we developed a **Hybrid Deep Learning Model** that integrates **1D Convolutional Neural Networks (CNN)** with **Transformer Encoders**. The CNN layers are designed to extract local geometric features (e.g., sharp price spikes and consolidation shapes), while the Transformer component utilizes Multi-Head Attention to capture long-range temporal dependencies within the time series. We also implemented a Baseline Bidirectional LSTM and an Ensemble model for performance benchmarking.

#### 1. Data Pipeline & Preprocessing
The workflow begins with an automated data ingestion process. The system downloads raw historical market data (CSV format) directly from the **shared drive**.
* **Filtering:** The pipeline strictly filters the dataset to retain only high-priority assets: **EURUSD** (Forex) and **XAU** (Gold).
* **Labeling:** A "Ground Truth" generation algorithm identifies consolidation zones preceded by strong trends ("poles"), labeling them as specific patterns (e.g., Bullish Flag).
* **Formatting:** The continuous time series data is transformed into fixed-size sliding window sequences (Length: 100) and normalized via MinMax scaling.

#### 2. Model Architectures
We developed and evaluated three distinct model architectures to solve this classification task:
* **Baseline Model (Bidirectional LSTM):** A standard Recurrent Neural Network serving as a benchmark.
* **Hybrid Model (CNN + Transformer):** The primary solution. This architecture combines CNNs for local feature extraction with Transformer Encoders to capture global context and long-range dependencies.
* **Ensemble Model:** A fusion architecture combining feature vectors from both LSTM and Hybrid branches.

#### 3. Hyperparameter Optimization
Extensive experiments were conducted (documented in `notebook/Tests.ipynb` and in deleted notebooks that can be found in history) to tune critical hyperparameters, including learning rate, batch size, CNN filter sizes, and Transformer attention heads. The configuration in `config.py` represents the optimal set derived from these tests.

#### 4. Results & Analysis
Evaluation was performed on a hold-out test set comprising 83 samples. The results highlight the complexity of the task and the impact of data quality:

* **Quantitative Results:** The **Hybrid Model** achieved an overall accuracy of **37%** with a weighted F1-score of **0.38**. While these numbers appear low, a deeper look at the metrics reveals promising behaviors:
    * **Directional Stability:** The model shows high precision in identifying the primary trend. For example, **Bullish Normal** patterns achieved a precision of **73%**.
    * **Trend vs. Shape:** The Confusion Matrix reveals that the model rarely confuses market directions (e.g., it rarely predicts a Bearish class for a Bullish input). The errors are concentrated within the *sub-classes* (e.g., misclassifying a "Bullish Normal" as a "Bullish Wedge"). Specifically, 10 instances of "Bullish Normal" were predicted as "Bullish Wedge," indicating that the model perceives the trend correctly but struggles to distinguish the subtle geometric differences defined in the labels.

* **Challenges & Limitations:**
    The performance is heavily constrained by data inherent issues rather than model capacity:
    * **Noisy Ground Truth:** The "Ground Truth" labels are generated algorithmically and are statistically noisy. Financial patterns are subjective; the distinction between a "Normal Flag" and a "Wedge" is often ambiguous even for human experts. This ambiguity prevents the model from learning sharp decision boundaries between shapes.
    * **Class Imbalance:** The dataset is highly unbalanced. As seen in the support metrics, "Normal" patterns (Support: 22-26) are far more frequent than specific shapes like "Pennants" (Support: 6-8). Although Focal Loss was implemented, the extreme scarcity of minority classes makes it difficult for the model to generalize on types like the *Bearish Pennant*.

* **Inference:** To mitigate the low raw accuracy, the final inference engine utilizes a strict confidence threshold (>80%). This filter suppresses uncertain predictions (where the model confuses the shape) and ensures that the system only alerts the user when it detects a high-probability pattern.
### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command. You must mount your local data directory to `/app/data` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run --rm \
  -v "$(pwd)/data":/app/data \
  -v "$(pwd)/output":/app/output \
  -v "$(pwd)/log":/app/log \
  dl-project > log/run.log 2>&1
```

### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for downloading, loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04-inference.py`: Script for running the model on new, unseen data to generate predictions.
    - ❗`05-cleanup.py`: Script for deleting the **data/** folder content (except for `ground_truth_labels.csv`) after everything finished
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-data-exploration.ipynb`: Notebook for initial exploratory data analysis (EDA) and visualization.
    - `02-label-analysis.ipynb`: Notebook for analyzing the distribution and properties of the target labels.
    - `03-hyperparameter-tuning.ipynb`: Notebook for tuning the hyperparameters and investigating the model architectures

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.

### Inference Input Configuration

The inference pipeline (`src/04-inference.py`) requires a **direct download link** to a CSV file (or a ZIP containing CSVs) specified in `src/config.py` via the `INFERENCE_URL` variable.

**⚠️ Important:** The link must trigger an immediate download. "Preview" or "View" links (HTML pages) will cause errors.

**Supported Link Formats:**
* **Direct Server Link:** `https://example.com/data/market_data.csv`
* **Google Drive (Direct):** Must use the export format:
    * `https://drive.google.com/uc?export=download&id=YOUR_FILE_ID`
* **SharePoint/OneDrive:** Must append the download query:
    * `https://.../sharepoint.com/.../file?download=1`

**❌ Do NOT use:**
* Folder view links (e.g., Google Drive folder view).
* HTML Preview links (e.g., `drive.google.com/file/d/.../view`).