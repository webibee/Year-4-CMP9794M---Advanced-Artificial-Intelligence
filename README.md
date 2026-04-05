# Bayesian Networks vs Gaussian Processes for Medical Diagnosis
- Course: CMP9794M – Advanced Artificial Intelligence
- Institution: University of Lincoln
- Grade Received: 70%
- Language: Python
- Libraries used: OpenCV, Standard Template Library (STL), Winsock2 (Windows Sockets)

# Overview
This project implements and compares two probabilistic machine learning approaches for medical diagnosis:

## Task 1: Bayesian Networks (50%)
- Exact inference – Inference by enumeration (Norvig & Russell, 2016)
- Approximate inference – Rejection sampling
- Parameter learning – Maximum Likelihood Estimation (MLE)
- Structure – Predefined Naive Bayes + configurable structures

## Task 2: Gaussian Processes (50%)
- Classification via Gaussian Processes – Mean vector + covariance matrix
- Regression-based training – Marginal likelihood maximisation
- Computational complexity – O(n³) exact inference
  
# Key Features
Bayesian Networks (Task 1)
| Component | Implementation |
|--------|-----------------|
| Structure | Predefined Naive Bayes (config file) + structure learning capability | 
| Parameter Learning | Maximum Likelihood Estimation with Laplace smoothing (+1 to avoid zero division) | 
| Exact Inference | Inference by enumeration (Norvig & Russell, 2016, pp. 523-524) | 
| Approximate Inference | Rejection sampling (1 million samples, incremental power-of-10 testing) | 
| Discretisation | Provided code for continuous → discrete feature conversion | 
	
	
Gaussian Processes (Task 2)
| Component | Implementation |
|--------|-----------------|
| Model | Gaussian Process Classification (Williams & Barber, 1998) | 
| Training | Mean vector + covariance matrix via regression | 
| Prediction | Normalised outputs → percentage probabilities | 
| Limitations | O(n³) complexity, memory constraints for large datasets | 

	
	
Evaluation Metrics
| Metric | Description | Target |
|--------|-----------------|------------|
| Balanced Accuracy | Average of sensitivity and specificity | Higher is better |
| F1 Score | Harmonic mean of precision and recall | Higher is better |
| AUC (Area Under Curve) | Overall classification performance | >0.5 = better than random |
| Brier Score | Mean squared difference between predicted and actual | Closer to 0 = better |
| KL Divergence | Statistical distance between probability distributions | Lower = more similar |
| Running Time | Training + inference time (seconds) | Lower = faster |

# How to run
# Prerequisites
```pip
pip install numpy pandas scikit-learn matplotlib
```

# Run Bayesian Network Inference
```pip
python BayesNetInference.py --config config-diabetes.txt --query "P(outcome=0|glucose=109, bmi=25.4, age=25)" --inference enumeration
```

# Run Model Evaluation
```pip
python ModelEvaluator.py --train diabetes_data-discretized-train.csv --test diabetes_data-discretized-test.csv --config config-diabetes.txt
```

# Run Gaussian Process Classification
```pip
python GaussianProcess.py --train diabetes_data.csv --test diabetes_data-test.csv --kernel RBF
```

		
		
		
		
		
		
