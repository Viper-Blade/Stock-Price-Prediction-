# Stock Price Prediction

## Overview

This project leverages advanced machine learning and deep learning models to forecast stock prices using time series data. It combines robust data preprocessing, feature engineering, and state-of-the-art architectures to deliver accurate and reliable predictions for financial forecasting.

### Key Features

- **End-to-end pipeline:** From raw data ingestion to final prediction and visualization.
- **Multiple model support:** XGBoost, LSTM, and WGAN-GP for comparative analysis.
- **Synthetic data generation:** Use GANs to augment datasets and improve model generalization.
- **Modular codebase:** Easy to extend, maintain, and adapt for other time series forecasting tasks.
- **Automated evaluation:** RMSE and visual plots for quick model assessment.

### Technologies Used

- Python 3.x
- NumPy, Pandas, Scikit-learn
- XGBoost
- TensorFlow/Keras (for LSTM)
- PyTorch (for WGAN-GP)
- Matplotlib, Seaborn (visualization)

## Project Architecture

The project is structured into several key modules:

1. **Data Preprocessing**

   - **Normalization:** Standardizes features for optimal model convergence.
   - **Dataset Splitting:** Separates data into training, validation, and test sets.
   - **Feature Engineering:** Uses Fourier transforms to extract periodic patterns and trends.

2. **Modeling**

   - **XGBoost:** Gradient boosting for tabular and time series data.
   - **LSTM:** Recurrent neural network for capturing temporal dependencies.
   - **WGAN-GP:** GAN architecture for generating realistic synthetic stock price data.

3. **Hyperparameter Optimization**

   - **GridSearchCV:** Exhaustive search for optimal XGBoost parameters.
   - **Custom GAN Training Loops:** Alternating optimization, learning rate scheduling, and gradient penalty for stable GAN training.

4. **Evaluation & Visualization**
   - **Metrics:** RMSE and other error metrics for model assessment.
   - **Plots:** Visual comparison of predicted vs. actual prices.

## Project Structure

```
Stock-Price-Prediction/
├── data/                # Raw and processed datasets
├── notebooks/           # Jupyter notebooks for exploration and prototyping
├── src/                 # Source code for models, preprocessing, and utilities
│   ├── preprocessing.py # Data cleaning and feature engineering
│   ├── models.py        # Model definitions (XGBoost, LSTM, WGAN-GP)
│   ├── train.py         # Training scripts and loops
│   └── evaluate.py      # Evaluation and visualization
├── results/             # Output plots and metrics
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## How It Works

1. **Data Preparation:** Load and preprocess stock price data, normalize features, and engineer new ones using Fourier transforms.
2. **Model Training:** Train XGBoost, LSTM, and WGAN-GP models on the processed data. Optimize hyperparameters for best performance.
3. **Evaluation:** Assess models using RMSE and visualize predictions against actual prices.
4. **Synthetic Data Generation:** Use WGAN-GP to generate realistic stock price sequences for data augmentation and analysis.

### Data Flow

1. **Raw Data** → 2. **Preprocessing & Feature Engineering** → 3. **Model Training** → 4. **Prediction & Evaluation** → 5. **Visualization & Reporting**

### Model Comparison Table

| Model   | Description                                    | RMSE (Lower is better) |
| ------- | ---------------------------------------------- | ---------------------- |
| XGBoost | Gradient boosting for tabular/time series data | **RMSE: 2.15**         |
| LSTM    | Deep RNN for sequential dependencies           | **RMSE: 1.87**         |
| WGAN-GP | GAN for synthetic data generation              | **RMSE: 2.03**         |

> **Note:** RMSE values are based on the test set and may vary with different datasets or hyperparameters.

## Results

### LSTM

<img width="408" alt="LSTM Result" src="https://github.com/user-attachments/assets/afd41c23-85b7-44c0-99d0-77df75a39552">

### XGBoost

<img width="452" alt="XGBoost Result" src="https://github.com/user-attachments/assets/86997036-5e7f-4ca6-9945-35f2ec4bcef7">

### WGAN-GP

<img width="452" alt="WGAN-GP Result" src="https://github.com/user-attachments/assets/72fe858a-d595-4fa7-a83f-02b2efe2aff3">

#### RMSE Scores

- **LSTM:** 1.87
- **XGBoost:** 2.15
- **WGAN-GP:** 2.03

These scores reflect the models' performance on the test set, demonstrating the effectiveness of deep learning and ensemble methods for time series forecasting.

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/S-Sharvesh/Stock-Price-Prediction.git
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the training scripts:**
   ```sh
   python src/train.py
   ```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## FAQ

**Q: Can I use this project for other time series data?**
A: Yes! The modular design allows you to adapt the pipeline for other forecasting tasks (e.g., weather, sales).

**Q: How do I add a new model?**
A: Implement your model in `src/models.py` and update the training/evaluation scripts accordingly.

**Q: What if I have questions or need help?**
A: Open an issue on GitHub or contact the author via the repository.

## License

This project is licensed under the MIT License.
