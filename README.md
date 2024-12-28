# House Price Prediction Using Machine Learning

## Overview
This project aims to predict house prices in India using various machine learning algorithms. The model is trained on a dataset containing various features that influence housing prices, providing insights for potential buyers and real estate investors.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Flask (for web application)

## Installation
To get started with this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/House-Price-Prediction.git
cd House-Price-Prediction
pip install -r requirements.txt
```

## Usage
Run the Flask application to start the web interface:

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your web browser to access the application.

## Dataset
The dataset used for training the model is sourced from [Kaggle](https://www.kaggle.com/datasets/rohanrao/house-prices-in-india). It includes various features such as:
- Location
- Size (in square feet)
- Number of bedrooms
- Age of the property
- Amenities

## Model Training
The project implements various machine learning algorithms, including:
- Linear Regression
- Decision Trees
- Random Forest
- XGBoost

The models are evaluated based on their performance metrics such as RMSE, MAE, and R² score.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.