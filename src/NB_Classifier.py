#############################################################################
# NB_Classifier.py
#
# Implements the Naive Bayes classifier for simple probabilistic inference.
# It assumes the existance of data in CSV format, where the first line contains
# the names of random variables -- the last being the variable to predict.
# This implementation aims to be agnostic of the data (no hardcoded vars/probs)
#
# This program is able to handle both discrete and continuous data, which is 
# automatically detected by updating flag continuous_inputs (initialised to False). 
#
# Version: 1.0, Date: 03 October 2022
# Version: 1.1, Date: 03 October 2023 more compatible with CPT_Generator
# Version: 1.2, Date: 08 October 2023 more compatible with ModelEvaluator
# Version: 1.3, Date: 01 November 2023 support for Gaussian distributions
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import math
import time
import numpy as np


class NB_Classifier:
    rand_vars = [] # names of random variables
    rv_key_values = {} # values of random variables
    rv_all_values = [] # data points/instances of the dataset
    predictor_variable = None # a.k.a. 'target variable'
    num_data_instances = 0 # number of data points/instances
    default_missing_count = 0.000001 # to avoid zero probabilities
    probabilities = {} # parameters of discrete Naive Bayes models
    gaussian_means = {} # parameters of Gaussian Naive Bayes models
    gaussian_stdevs = {} # parameters of Gaussian Naive Bayes models
    predictions = [] # probabilistic predictions for rv_all_values
    log_probabilities = False # log probabilities are negative values
    continuous_inputs = False # flag to set discrete or continuous data
    stardardise_data = True # use standardised data instead of raw data
    verbose = False

    def __init__(self, file_name, fitted_model=None):
        if file_name is None:
            return
        else:
            self.read_data(file_name)

        if fitted_model is None:
            self.training_time = time.time()
            if self.continuous_inputs:
                self.estimate_means_and_standard_deviations()
            else:
                self.estimate_probabilities()
            self.training_time = time.time() - self.training_time

        else:
            self.inference_time = time.time()
            self.rv_key_values = fitted_model.rv_key_values
            self.probabilities = fitted_model.probabilities
            self.training_time = fitted_model.training_time
            self.test_learnt_probabilities(file_name)
            self.inference_time = time.time() - self.inference_time

    def read_data(self, data_file):
        print("\nREADING data file %s..." % (data_file))
        print("---------------------------------------")

        self.rand_vars = []
        self.rv_key_values = {}
        self.rv_all_values = []

        with open(data_file) as csv_file:
            for line in csv_file:
                line = line.strip()
                line = line.replace('ï»¿', '')
                line = line.replace('∩╗┐', '')

                if len(self.rand_vars) == 0:
                    self.rand_vars = line.split(',')
                    print("VARS="+str(self.rand_vars))
                    for variable in self.rand_vars:
                        self.rv_key_values[variable] = []
                else:
                    values = line.split(',')

                    if len(self.rv_all_values) == 0:
                        self.continuous_inputs = self.check_datatype(values)
                        print("self.continuous_inputs="+str(self.continuous_inputs))

                    if self.continuous_inputs is True:
                        values = [float(value) for value in values]

                    self.rv_all_values.append(values)
                    self.update_variable_key_values(values)
                    self.num_data_instances += 1

        if self.stardardise_data is True and self.continuous_inputs and True:
            self.rv_all_values = self.standardise_data(self.rv_all_values, data_file)

        self.predictor_variable = self.rand_vars[len(self.rand_vars)-1]

        print("RANDOM VARIABLES=%s" % (self.rand_vars))
        print("VARIABLE KEY VALUES=%s" % (self.rv_key_values))
        print("VARIABLE VALUES (first 10)=%s" % (self.rv_all_values[:10]))
        print("PREDICTOR VARIABLE=%s" % (self.predictor_variable))
        print("|data instances|=%d" % (self.num_data_instances))

    def standardise_data(self, X, datafile):
        print("NORMALISING inputs of datafile=%s..." % (datafile))
        X = np.asarray(X)
        X_normalised = np.zeros(X.shape)
        for i in range(0, len(X[0])):
            if i == len(X[0])-1:
                X_normalised[:,i] = X[:,i].astype(int)
            else:
                X_column_i = X[:,i]
                _mean = np.mean(X_column_i)
                _std = np.std(X_column_i)
                X_normalised[:,i] = (X_column_i-_mean)/(_std)
        print("X_normalised=",X_normalised)
        return X_normalised

    def check_datatype(self, values):
        for feature_value in values:
            if feature_value[0].isalpha():
                return False # discrete data (due to values being alphabetic characters)
            elif len(feature_value.split('.')) > 1 or len(feature_value) > 1:
                return True # continuous data (due to decimals or values above digits)
        return False # discrete data (due to not finding decimals and only digits)

    def update_variable_key_values(self, values):
        for i in range(0, len(self.rand_vars)):
            variable = self.rand_vars[i]
            key_values = self.rv_key_values[variable]
            value_in_focus = values[i]
            if value_in_focus not in key_values:
                self.rv_key_values[variable].append(value_in_focus)

    def estimate_probabilities(self):
        countings = self.estimate_countings()
        prior_counts = countings[self.predictor_variable]

        print("\nESTIMATING probabilities...")
        for variable, counts in countings.items():
            prob_distribution = {}
            for key, val in counts.items():
                variables = key.split('|')

                if len(variables) == 1:
                    # prior probability
                    probability = float(val/self.num_data_instances)
                else:
                    # conditional probability
                    probability = float(val/prior_counts[variables[1]])

                if self.log_probabilities is False:
                    prob_distribution[key] = probability
                else:
                    # convert probability to log probability
                    prob_distribution[key] = math.log(probability)

            self.probabilities[variable] = prob_distribution

        for variable, prob_dist in self.probabilities.items():
            prob_mass = 0
            for value, prob in prob_dist.items():
                prob_mass += prob
            print("P(%s)=>%s\tSUM=%f" % (variable, prob_dist, prob_mass))

    def estimate_countings(self):
        print("\nESTIMATING countings...")

        countings = {}
        for variable_index in range(0, len(self.rand_vars)):
            variable = self.rand_vars[variable_index]

            if variable_index == len(self.rand_vars)-1:
                # prior counts
                countings[variable] = self.get_counts(None)
            else:
                # conditional counts
                countings[variable] = self.get_counts(variable_index)

        print("countings="+str(countings))
        return countings

    def get_counts(self, variable_index):
        counts = {}
        predictor_index = len(self.rand_vars)-1

        # accumulate countings
        for values in self.rv_all_values:
            if variable_index is None:
                # case: prior probability
                value = values[predictor_index]
            else:
                # case: conditional probability
                value = values[variable_index]+"|"+values[predictor_index]

            try:
                counts[value] += 1
            except Exception:
                counts[value] = 1

        # verify countings by checking missing prior/conditional counts
        if variable_index is None:
            counts = self.check_missing_prior_counts(counts)
        else:
            counts = self.check_missing_conditional_counts(counts, variable_index)

        return counts

    def check_missing_prior_counts(self, counts):
        for var_val in self.rv_key_values[self.predictor_variable]:
            if var_val not in counts:
                print("WARNING: missing count for variable=" % (var_val))
                counts[var_val] = self.default_missing_count

        return counts

    def check_missing_conditional_counts(self, counts, variable_index):
        variable = self.rand_vars[variable_index]
        for var_val in self.rv_key_values[variable]:
            for pred_val in self.rv_key_values[self.predictor_variable]:
                pair = var_val+"|"+pred_val
                if pair not in counts:
                    print("WARNING: missing count for variables=%s" % (pair))
                    counts[pair] = self.default_missing_count

        return counts

    def test_learnt_probabilities(self, file_name):
        print("\nEVALUATING on "+str(file_name))

        # iterates over all instances in the test data
        for instance in self.rv_all_values:
            distribution = {}
            if self.verbose:
                print("Input vector=%s" % (instance))

            # iterates over all values in the predictor variable
            for predictor_value in self.rv_key_values[self.predictor_variable]:
                prob_dist = self.probabilities[self.predictor_variable]
                prob = prob_dist[predictor_value]

                # iterates over all instance values except the predictor var.
                for value_index in range(0, len(instance)-1):
                    variable = self.rand_vars[value_index]
                    x = instance[value_index]
                    if self.continuous_inputs is False:
                        prob_dist = self.probabilities[variable]
                        cond_prob = x+"|"+predictor_value

                        if self.log_probabilities is False:
                            prob *= prob_dist[cond_prob]
                        else:
                            prob += prob_dist[cond_prob]
                    else:
                        # this block of code has only been tested with non-log probabilities
                        probA = self.get_probability_density(x, variable, predictor_value)
                        probB = self.get_probability_density(x, variable, abs(1-predictor_value))
                        unnormalised_dist = {'predictor_value':probA, 'nonpredictor_value':probB}
                        prob_dist = self.get_normalised_distribution(unnormalised_dist)
                        probability = prob_dist['predictor_value']
                        prob *= probability

                distribution[predictor_value] = prob

            normalised_dist = self.get_normalised_distribution(distribution)
            self.predictions.append(normalised_dist)
            if self.verbose:
                print("UNNORMALISED DISTRIBUTION=%s" % (distribution))
                print("NORMALISED DISTRIBUTION=%s" % (normalised_dist))
                print("---")

    def get_normalised_distribution(self, distribution):
        normalised_dist = {}
        prob_mass = 0
        for var_val, prob in distribution.items():
            prob = math.exp(prob) if self.log_probabilities is True else prob
            prob_mass += prob

        for var_val, prob in distribution.items():
            prob = math.exp(prob) if self.log_probabilities is True else prob
            normalised_prob = prob/prob_mass
            normalised_dist[var_val] = normalised_prob

        return normalised_dist

    def estimate_means_and_standard_deviations(self):
        print("\nCALCULATING means and standard deviations...")

        # iterate over all random variables except the predictor var.
        for value_index in range(0, len(self.rand_vars)-1):
            variable = self.rand_vars[value_index]
            print("variable="+str(variable))

            # iterate over all training instances gather feature vectors
            feature_vectors = {}
            for instance in self.rv_all_values:
                predictor_value = instance[len(instance)-1]
                value = instance[value_index]
                if predictor_value not in feature_vectors:
                    feature_vectors[predictor_value] = []
                feature_vectors[predictor_value].append(value)

            # calculate means and standard deviations from the feature vectors
            self.gaussian_means[variable] = {}
            self.gaussian_stdevs[variable] = {}
            for predictor_value in feature_vectors:
                mean = np.mean(feature_vectors[predictor_value])
                stdev = np.std(feature_vectors[predictor_value])
                self.gaussian_means[variable][predictor_value] = mean
                self.gaussian_stdevs[variable][predictor_value] = stdev

            print("\tmeans="+str(self.gaussian_means[variable]))
            print("\tstdevs="+str(self.gaussian_stdevs[variable]))

        # compute prior probabilities of the predictor variable
        prior_distribution = {}
        print("self.predictor_variable="+str(self.predictor_variable))
        print("self.rv_key_values="+str(self.rv_key_values))
        for predictor_value in self.rv_key_values[self.predictor_variable]:
            prob = len(feature_vectors[predictor_value])/len(self.rv_all_values)
            prior_distribution[predictor_value] = prob
        self.probabilities[self.predictor_variable] = prior_distribution
        print("priors="+str(self.probabilities[self.predictor_variable]))

    def get_probability_density(self, x, variable, predictor_value):
        mean = self.gaussian_means[variable][predictor_value]
        stdev = self.gaussian_stdevs[variable][predictor_value]
        e_val = -0.5*np.power((x-mean)/stdev, 2)
        probability = (1/(stdev*np.sqrt(2*np.pi))) * np.exp(e_val)
        return probability



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: NB_Classifier.py [train_file.csv] [test_file.csv]")
        exit(0)
    else:
        file_name_train = sys.argv[1]
        file_name_test = sys.argv[2]
        nb_fitted = NB_Classifier(file_name_train)
        nb_tester = NB_Classifier(file_name_test, nb_fitted)