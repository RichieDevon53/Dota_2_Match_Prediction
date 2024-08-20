# Dota 2 Team Lineup Prediction Using Machine Learning
### Introduction
This project aims to analyze and predict the potential success of a Dota 2 team based on the heroes they select, using machine learning models. The analysis is based on a dataset from Tier 1 Dota 2 tournaments spanning from 2019 to 2022. The project is designed to assist a professional esports team in optimizing their strategies and improving their chances of winning in upcoming international tournaments.

### Objective
The primary objective is to develop predictive models that can forecast the outcomes of matches based on the team compositions used. By analyzing historical data, the project aims to identify the most effective strategies and hero combinations that can lead to victory

### Background
As a Data Analyst for a professional esports team, the need to leverage data from past tournaments is crucial for predicting optimal hero picks and strategies. Dota 2, a complex multiplayer online battle arena (MOBA) game, requires teams to carefully choose heroes that complement each other and counter the opponent's lineup. This project helps in providing actionable insights to the team management and coaching staff, enabling them to prepare better for each match.

### Why Dota 2?
Dota 2 is a globally popular MOBA game developed by Valve Corporation, known for its strategic depth and highly competitive scene. With tournaments like The International offering multi-million dollar prize pools, the stakes are incredibly high. Predicting hero picks and outcomes can give teams a significant edge in such a competitive environment.

### Dataset
The dataset consists of match information from Tier 1 Dota 2 tournaments held between 2019 and 2022. It includes details about the heroes picked, match outcomes, and various in-game statistics that are essential for training the predictive models.

### Analysis Process
1. Data Collection: Tournament data from 2019 to 2022 was gathered, focusing on Tier 1 events.
2. Data Cleaning: The dataset was cleaned to remove inconsistencies, missing values, and outliers.
3. Feature Engineering: Relevant features such as hero synergy, counter picks, and match outcomes were created.
4. Model Development: Various machine learning models were developed and evaluated to predict match outcomes based on hero picks.
5. Model Evaluation: The models were evaluated for accuracy, precision, and recall to ensure reliability.
6. Deployment: The best-performing model is deployed for real-time predictions to assist in strategic planning.

### Tools and Libraries
- Python: The primary programming language used for analysis.
- Pandas: For data manipulation and analysis.
- Scikit-learn: For building and evaluating machine learning models.
- XGBoost: For advanced model development.
- Matplotlib & Seaborn: For data visualization.
- Jupyter Notebook: For documenting the analysis process.

### Model Deployment
The trained model is deployed on <a href='https://huggingface.co/spaces/53Devon/Dota_2_Machine_Learning?logs=container'>Hugging Face Spaces</a>, where it can be accessed for real-time predictions.

### Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.
