# BULL FLAG DETECTOR PROJECT by Kránitz Bence
As an Applied Mathematics student, my experience has focused on data science projects in Jupyter notebooks. This is my first project at this scale and my initial venture into Docker. **The models were trained on CPU, so they are intentionally designed for efficiency rather than maximum complexity.**


## Submission Instructions

### Submission Checklist

Before submitting your project, ensure you have completed the following steps.
**Please note that the submission can only be accepted if these minimum requirements are met.**

- [x] **Project Information**: Filled out the "Project Information" section (Topic, Name, Extra Credit).
- [x] **Solution Description**: Provided a clear description of your solution, model, and methodology.
- [x] **Extra Credit**: If aiming for +1 mark, filled out the justification section.
- [x] **Data Preparation**: Included a script or precise description for data preparation.
- [x] **Dependencies**: Updated `requirements.txt` with all necessary packages and specific versions.
- [x] **Configuration**: Used `src/config.py` for hyperparameters and paths, contains at least the number of epochs configuration variable.
- [x] **Logging**:
    - [x] Log uploaded to `log/run.log`
    - [x] Log contains: Hyperparameters, Data preparation and loading confirmation, Model architecture, Training metrics (loss/acc per epoch), Validation metrics, Final evaluation results, Inference results.
- [ ] **Docker**:
    - [ ] `Dockerfile` is adapted to your project needs.
    - [ ] Image builds successfully (`docker build -t dl-project .`).
    - [ ] Container runs successfully with data mounted (`docker run ...`).
    - [ ] The container executes the full pipeline (preprocessing, training, evaluation).
- [ ] **Cleanup**:
    - [ ] Removed unused files.
    - [ ] **Deleted this "Submission Instructions" section from the README.**

## Project Details

### Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: Kránitz Bence
- **Aiming for +1 Mark**: No

### Solution Description

This project addresses the challenge of automating technical analysis in financial markets, specifically the detection of complex chart patterns such as Bullish and Bearish Flags, Pennants, and Wedges. Manual identification of these patterns is subjective, time-consuming, and difficult to scale across multiple assets like EURUSD and Gold (XAU).

To solve this, we developed a **Hybrid Deep Learning Model** that integrates **1D Convolutional Neural Networks (CNN)** with **Transformer Encoders**. The CNN layers are designed to extract local geometric features (e.g., sharp price spikes and consolidation shapes), while the Transformer component utilizes Multi-Head Attention to capture long-range temporal dependencies within the time series. We also implemented a Baseline Bidirectional LSTM and an Ensemble model for performance benchmarking.

#### 1. Data Pipeline & Preprocessing
The workflow begins with an automated data ingestion process. The system downloads raw historical market data (CSV format) directly from a secure shared drive.
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
[Adjust the commands that show how do build your container and run it with log output.]

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command. You must mount your local data directory to `/app/data` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run -v /absolute/path/to/your/local/data:/app/data dl-project > log/run.log 2>&1
```

*   Replace `/absolute/path/to/your/local/data` with the actual path to your dataset on your host machine that meets the [Data preparation requirements](#data-preparation).
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).


### File Structure and Functions

[Update according to the final file structure.]

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04-inference.py`: Script for running the model on new, unseen data to generate predictions.
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