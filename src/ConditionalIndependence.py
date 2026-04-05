#############################################################################
# ConditionalIndependence.py
#
# Implements functionality for conditional independence tests via the
# library causal-learn (https://github.com/cmu-phil/causal-learn), which
# can be used to identify edges to keep or remove in a graph given a dataset.
# The flag 'chi_square_test' can be used to change tests between X^2 and G^2.
# The flag use_continuous_data can be used to change test to fisherz.
#
# This requires installing the following (at Uni-Lincoln computer labs):
# 1. Type Anaconda Prompt in your Start icon
# 2. Open your terminal as administrator
# 3. Execute=> pip install causal-learn
#
# USAGE instructions to run this program can be found at the bottom of this file.
#
# Version: 1.0, Date: 19 October 2022 (first version)
# Version: 1.1, Date: 07 October 2023 (minor revision)
# Version: 1.2, Date: 03 October 2023 (support for continuous data)
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import numpy as np
from causallearn.utils.cit import CIT


class ConditionalIndependence:
    chisq_obj = None
    rand_vars = []
    rv_all_values = []
    chi_square_test = True
    use_continuous_data = False

    def __init__(self, file_name):
        data = self.read_data(file_name)
        print(data)
        test = "chisq" if self.chi_square_test else "gsq"
        test = "fisherz" if self.use_continuous_data else test
        self.chisq_obj = CIT(data, test)

    def read_data(self, data_file):
        print("\nREADING data file %s..." % (data_file))
        print("---------------------------------------")

        with open(data_file) as csv_file:
            for line in csv_file:
                line = line.strip()
                if len(self.rand_vars) == 0:
                    self.rand_vars = line.split(',')
                else:
                    values = line.split(',')
                    if self.use_continuous_data:
                        for i in range(0, len(values)):
                            values[i] = float(values[i])
                    self.rv_all_values.append(values)

        print("RANDOM VARIABLES=%s" % (self.rand_vars))
        print("VARIABLE VALUES (first 10)=%s" % (self.rv_all_values[:10])+"\n")
        if self.use_continuous_data:
            return np.array(self.rv_all_values)
        else:
            return self.rv_all_values

    def parse_test_args(self, test_args):
        main_args = test_args[2:len(test_args)-1]
        variables = main_args.split('|')[0]
        Vi = variables.split(',')[0]
        Vj = variables.split(',')[1]
        parents_i = []
        for parent in (main_args.split('|')[1].split(',')):
            if parent.lower() == 'none':
                continue
            else:
                parents_i.append(parent)

        return Vi, Vj, parents_i

    def get_var_index(self, target_variable):
        for i in range(0, len(self.rand_vars)):
            if self.rand_vars[i] == target_variable:
                return i
        print("ERROR: Couldn't find index of variable "+str(target_variable))
        return None

    def get_var_indexes(self, parent_variables):
        if len(parent_variables) == 0:
            return None
        else:
            index_vector = []
            for parent in parent_variables:
                index_vector.append(self.get_var_index(parent))
            return index_vector

    def compute_pvalue(self, variable_i, variable_j, parents_i):
        var_i = self.get_var_index(variable_i)
        var_j = self.get_var_index(variable_j)
        par_i = self.get_var_indexes(parents_i)
        p = self.chisq_obj(var_i, var_j, par_i)

        print("X2test: Vi=%s, Vj=%s, pa_i=%s, p=%s" %
              (variable_i, variable_j, parents_i, p))
        return p


if len(sys.argv) != 3:
    print("USAGE: ConditionalIndepencence.py [train_file.csv] [I(Vi,Vj|parents)]")
    print("EXAMPLE1: python ConditionalIndependence.py lang_detect_train.csv \"I(X1,X2|Y)\“")
    print("EXAMPLE2: python ConditionalIndependence.py lang_detect_train.csv \"I(X1,X15|Y)\“")
    exit(0)
else:
    data_file = sys.argv[1]
    test_args = sys.argv[2]

    ci = ConditionalIndependence(data_file)
    Vi, Vj, parents_i = ci.parse_test_args(test_args)
    ci.compute_pvalue(Vi, Vj, parents_i)
