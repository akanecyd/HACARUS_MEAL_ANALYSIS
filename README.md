# HACARUS_MEAL_ANALYSIS

This project is dedicated to developing a machine learning model to analyze and predict the quality of meals. The primary goal is to distinguish between meals of varying quality levels, categorized as 'worst', 'bad', 'good', and 'best'.

The repository includes two main Python scripts:

Model_comparision.py: This script is used for comparing and selecting the most suitable model for meal quality prediction. It includes different machine learning models and evaluates their performance based on predefined metrics. The output is a ranked list of models in terms of their ability to predict meal quality.

Meal_analyze.py: Once the best model is selected using Model_comparision.py, this script is used to fit the selected model to the data and conduct further analysis. The script includes functions for training the model, making predictions, evaluating performance, and visualizing the results.

To use these scripts, follow the instructions below:

# Prerequisites
Ensure you have Python 3.x installed along with the following Python libraries:

pandas
numpy
scikit-learn
matplotlib


# Instructions
Clone the repository to your local machine using git clone.

Install necessary Python packages using pip: pip install -r requirements.txt.

Run Model_comparision.py to evaluate different models and select the most suitable one. Use the command python Model_comparision.py.

Once you have selected the best model, run Meal_analyze.py to train the model and conduct the analysis. Use the command python Meal_analyze.py.

# Contact
If you have any questions, feel free to reach out. Any suggestions for improvement are always welcome.
