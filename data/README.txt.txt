The files in this folder contain data for assessment item 1 of CMP9494M Advanced Artificial Intelligence 2023-24.
Two datasets were selected for this assessment. Here the original sources:
(1) https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
(2) https://www.kaggle.com/datasets/mathchi/diabetes-data-set

The following is the set of features for the cardiovascular data:
Age | Objective Feature | age | int (days)
Height | Objective Feature | height | int (cm) |
Weight | Objective Feature | weight | float (kg) |
Gender | Objective Feature | gender | categorical code |
Systolic blood pressure | Examination Feature | ap_hi | int |
Diastolic blood pressure | Examination Feature | ap_lo | int |
Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
Smoking | Subjective Feature | smoke | binary |
Alcohol intake | Subjective Feature | alco | binary |
Physical activity | Subjective Feature | active | binary |
Presence or absence of cardiovascular disease | Target Variable | target | binary |

The following is the set of features for the diabetes data:
Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
Outcome: Class variable (0 or 1)

Note that the CSV files may not use the full names of features (or random variables) but a short notation.
An example above is 'ap_hi', which refers to 'Systolic blood pressure'.

Since the original data contains continues values, each column (X in the code below) in a CSV file was
discretised using the following method.

def get_discretized_vector(X):
    _mean = np.mean(X)
    _std = np.std(X)
    bins = [_mean-(_std*2), _mean-(_std*1), _mean, _mean+(_std*1), _mean+(_std*2)]
    X_discretized = np.digitize(X, bins)
    return X_discretized

Last but not least, the discretised data was randomly split into training and test files with an 80:20 ratio.

The sizes of training/test examples are:
cardiovascular_data-discretized-train=55930
cardiovascular_data-discretized-test=14070
diabetes_data-discretized-train=603
diabetes_data-discretized-test=165

This folder includes the original data files as well as their discretised versions in case you want to compare them.