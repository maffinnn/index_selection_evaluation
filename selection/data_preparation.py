import sys
import csv
import ast
from csv import reader

maxint = int(sys.maxsize/10000000000)
csv.field_size_limit(maxint)

def read_csv(filename):
    with open(filename,'r') as rf:
        dreader = reader(rf)
        data_list = list(dreader)
    return data_list

# data_list[i][0]: Query ID and Query text for the i-th query
# data_list[i][1]: Index configurations for the i-th query
# data_list[i][2]: Average cost of each configuration for the i-th query
# data_list[i][3]: Query execution plan of each configuration for the i-th query
# data_list[i][4]: Details execution costs (each query is executed 4 times and the last 3 times are recorded) of each configuration for the i-th query

# To convert the index confituration string to list of index names
def index_conversion(indexes):    
    indexes_s = indexes[1:-1].split(', ')
    index_new = []
    start = 0
    for i in range(0,len(indexes_s)):
        if indexes_s[i][0] == '(':
            start = i
        elif len(indexes_s[i]) >2 and indexes_s[i][-2:] == '))':
            end = i
            index_new.append(','.join(indexes_s[start:end+1])[1:-1])
            start = 0
        elif start == 0:
            print(indexes_s[i])
            index_new.append(indexes_s[i])

    return index_new

# Example usage:
# indexes = index_conversion(data_list[0][1])
# print("index configuration: ", indexes)
# print("Average cost of each configuration: ", data_list[0][2])


# Convert the plans string to query plan dictionary

# plans = data_list[0][3]
# plan_ds = ast.literal_eval(plans)
# print(len(plan_ds))

