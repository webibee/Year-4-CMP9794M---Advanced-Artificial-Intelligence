#############################################################################
# PDF_Generator.py
#
# This program generates Conditional Probability Density Functions (PDFs) 
# into a config file in order to be useful for probabilistic inference. 
# Similarly to CPT_Generator, it does that by rewriting a given config file 
# without PDFs. The newly generated PDFs, derived from the given data file 
# (in CSV format), re-write the provided configuration file. They also 
# generate an additional file with extension .pkl containing regression 
# models, one per random variable in the Bayesian network. The purpose of
# the regression models is to predict the means of new data points, which
# can be used for probabilistic inference later on. This file focuses on
# estimating the parameters of a BayesNet with continuous inputs.
#
# At the bottom of the file is an example on how to run this program. 
#
# Version: 1.0, Date: 11 October 2022, first version using linear regression
# Version: 2.0, Date: 25 October 2023, version using (neural) non-linear regression
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import time
import pickle
import numpy as np
from BayesNetReader import BayesNetReader
from NB_Classifier import NB_Classifier
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neural_network import MLPRegressor


class PDF_Generator(BayesNetReader, NB_Classifier):
    bn = None # for instance of BayesNetReader
    nbc = None # for instance of NB_Classifier
    means = {} # mean vectors of random variables
    stdevs = {} # standard deviations of random variables
    regressors = {} # regression models for random variables

    def __init__(self, configfile_name, datafile_name):
        # Bayes net reading (configfile_name in text format)
        # and data loading (datafile_name in csv format)
        self.bn = BayesNetReader(configfile_name)
        self.nbc = NB_Classifier(None)
        self.nbc.read_data(datafile_name)

        # model training and saving, which updates the text file
        # configfile_name and generates an equivalent *.pkl file 
        self.running_time = time.time()
        self.estimate_regression_models()
        self.update_configuration_file(configfile_name)
        self.running_time = time.time() - self.running_time
        print("Running Time="+str(self.running_time)+" secs.")

    # computes the following for each random variable in the network:
	# (a) mean and standard deviation
    # (b) regression models via GradientBoostingRegressor,
    #     which you can change to the method of your choice.
    def estimate_regression_models(self):
        print("\nESTIMATING non-linear regression models...")
        print("---------------------------------------------------")

        for pd in self.bn.bn["structure"]:
            print(str(pd))
            p = pd.replace('(', ' ')
            p = p.replace(')', ' ')
            tokens = p.split("|")

            # estimate mean and standard deviation per random variable
            variable = tokens[0].split(' ')[1]
            feature_vector = self.get_feature_vector(variable)
            self.means[variable] = np.mean(feature_vector)
            self.stdevs[variable] = np.std(feature_vector)
            #print("mean=%s stdev=%s" % (mean, stdev))
            #exit(0)

            # train regression models via a GradientBoostingRegressor
            if len(tokens) == 2:
                variable = tokens[0].split(' ')[1]
                parents = tokens[1].strip().split(',')
                inputs, outputs = self.get_feature_vectors(parents, variable)
                regression_model = MLPRegressor(hidden_layer_sizes=(200,100,30), \
				                   max_iter=1000, activation='relu', early_stopping=True)
                #regression_model = LinearRegression()#Ridge()
                regression_model.fit(inputs, outputs)
                self.regressors[variable] = regression_model
                print("Created regression model for variable %s\n" % (variable))
                #exit(0)
            else:
                print("Estimated means and stdevs for variable %s\n" % (variable))

    # returns the data column of the random variable given as argument
    def get_feature_vector(self, variable):
        variable_index = self.get_variable_index(variable)
        feature_vector = []
        counter = 0
        for datapoint in self.nbc.rv_all_values:
            value = datapoint[variable_index]
            feature_vector.append(value)

        return np.asarray(feature_vector, dtype="float32")

    # returns the index (0 to N-1) of the random variable given as argument
    def get_variable_index(self, variable):
        for i in range(0, len(self.nbc.rand_vars)):
            if variable == self.nbc.rand_vars[i]:
                return i
        print("WARNING: couldn't find index of variables=%s" % (variable))
        return None

    # return the data columns of the parent random variables given as argument
    def get_feature_vectors(self, parents, variable):
        input_features = []
        for parent in parents:
            feature_vector = self.get_feature_vector(parent)
            if len(input_features) == 0:
                for f in range(0, len(feature_vector)):
                    input_features.append([feature_vector[f]])
            else:
                for f in range(0, len(feature_vector)):
                    tmp_vector = input_features[f]
                    tmp_vector.append(feature_vector[f])
                    input_features[f] = tmp_vector

        output_features = self.get_feature_vector(variable)

        input_features = np.asarray(input_features, dtype="float32")
        output_features = np.asarray(output_features, dtype="float32")
        return input_features, output_features

    # re-writes the provided configuration file with information about
    # regression models -- one per random variable in the network.
    # The means, standard deviations and regression models are all
    # stored in a PICKLE file due for data saving/loading convience.
    # Such files with extension .pkl are stored in the config folder.
    def update_configuration_file(self, configfile_name):
        print("WRITING config file with regression models...")
        print("See rewritten file "+str(configfile_name))
        print("---------------------------------------------------")
        name = self.bn.bn["name"]

        rand_vars = self.bn.bn["random_variables_raw"]
        rand_vars = str(rand_vars).replace('[', '').replace(']', '')
        rand_vars = str(rand_vars).replace('\'', '').replace(', ', ';')

        structure = self.bn.bn["structure"]
        structure = str(structure).replace('[', '').replace(']', '')
        structure = str(structure).replace('\'', '').replace(', ', ';')
		
        regression_models = {}
        regression_models['means'] = self.means
        regression_models['stdevs'] = self.stdevs
        regression_models['regressors'] = self.regressors
        regression_models_filename = configfile_name[:len(configfile_name)-4]
        regression_models_filename = regression_models_filename+'.pkl'

        with open(configfile_name, 'w') as cfg_file:
            cfg_file.write("name:"+str(name))
            cfg_file.write('\n')
            cfg_file.write('\n')
            cfg_file.write("random_variables:"+str(rand_vars))
            cfg_file.write('\n')
            cfg_file.write('\n')
            cfg_file.write("structure:"+str(structure))
            cfg_file.write('\n')
            cfg_file.write('\n')
            cfg_file.write("regression_models:"+str(regression_models_filename))

        with open(regression_models_filename, 'wb') as models_file:
            pickle.dump(regression_models, models_file)


if len(sys.argv) != 3:
    print("USAGE: PDF_Generator.py [your_config_file.txt] [training_file.csv]")
    print("EXAMPLE> PDF_Generator.py config_banknote_authentication.txt data_banknote_authentication-train.csv")
    exit(0)

else:
    configfile_name = sys.argv[1]
    datafile_name = sys.argv[2]
    PDF_Generator(configfile_name, datafile_name)
