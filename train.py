#!/home/datamount/biansiyuan/anaconda3/envs/py38/bin/python

import numpy as np
import pandas as pd
import ast
import re
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
from selection.data_preparation import read_csv, index_conversion
from selection.workload import Query, Index, Column, Table
from constant import TPC_DS_TABLE_PREFIX, TPC_H_TABLE_PREFIX, LOGICAL_OPERATORS, PHYSICAL_OPERATORS, PHYISCAL_TO_LOGICAL_OPERATOR_MAP
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

MAX_TABLE_NUM = 25
MAX_COLUMN_NUM = 429
MAX_CONFIG_LEN = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

def q_error(actual, pred):
    epsilon = 1e-4
    q_e = 0
    for i in range(len(pred)):
        q_e += max((actual[i]+epsilon)/(pred[i]+epsilon),(pred[i]+epsilon)/(actual[i]+epsilon))
    return q_e/len(pred)

def read_table_info(row_info_filepath, column_info_filepath):
    data_table_info = read_csv(row_info_filepath)[2:-2]
    table_dict = OrderedDict()
    for table_info in data_table_info:
        table_info_tuple = table_info[0].split('|')
        if len(table_info_tuple) < 2: continue
        table_name = table_info_tuple[0].strip()
        row_count = float(table_info_tuple[1].strip())
        table = Table(table_name)
        table.set_row_count(row_count)
        table_dict[table_name] = table
        
    TABLE_PREFIX_MAP = None
    if re.search("TPC_DS", column_info_filepath) or re.search("dsb", column_info_filepath):
        TABLE_PREFIX_MAP = TPC_DS_TABLE_PREFIX
    elif re.search("TPCH", column_info_filepath):
        TABLE_PREFIX_MAP = TPC_H_TABLE_PREFIX
    elif re.search("IMDB", column_info_filepath): 
        TABLE_PREFIX_MAP = None
    else: 
        raise ValueError("Specified dataset not supported")
    
    data_column_info = read_csv(column_info_filepath)[2:-2]
    column_dict = OrderedDict()
    for column_info in data_column_info:
        column_info_tuple = column_info[0].split('|')
        if len(column_info_tuple) < 3: continue
        
        if TABLE_PREFIX_MAP == None:
            ## IMDB
            table_name = column_info_tuple[0].strip()
            column_name = column_info_tuple[1].strip()
            null_frac = float(column_info_tuple[2].strip())
            n_distinct = float(column_info_tuple[3].strip())
            column = Column(column_name)
            column.set_cardinality(-n_distinct * row_count if n_distinct < 0 else n_distinct)
            column.set_null_fraction(null_frac)
            if table_name in table_dict.keys():
                column.table = table_dict[table_name]
                column.table.add_column(column)
        else:
            column_name = column_info_tuple[0].strip()
            null_frac = float(column_info_tuple[1].strip())
            n_distinct = float(column_info_tuple[2].strip())
            column = Column(column_name)
            column.set_cardinality(-n_distinct * row_count if n_distinct < 0 else n_distinct)
            column.set_null_fraction(null_frac)
            if (prefix := column_name.split('_')[0]) in TABLE_PREFIX_MAP.keys():
                table_name = TABLE_PREFIX_MAP[prefix]
                if table_name in table_dict.keys():
                    column.table = table_dict[table_name]
                    column.table.add_column(column)      
        column_dict[column_name] = column
    return table_dict, column_dict

def convert_configuration_to_obj(columns_dict, config_string):
    configs = []
    for config_s in config_string:
        if config_s == "[]": 
            configs.append([])
            continue
        config = []
        indexes_s = config_s.split('I')
        for index_s in indexes_s:
            if index_s == '': continue
            table_columns = index_s.split('C')
            indexed_columns = []
            for table_column in table_columns:
                table_column = table_column.strip('(), ')
                if table_column == '': continue
                column_name = table_column.split('.')[-1]
                if column_name in columns_dict:
                    indexed_columns.append(columns_dict[column_name])
            config.append(Index(indexed_columns))
        configs.append(config)
    return configs

def read_query_and_index_data(filepath, column_dict):
    # data_list[i][0]: Query ID and Query text for the i-th query
    # data_list[i][1]: Index configurations for the i-th query
    # data_list[i][2]: Average cost of each configuration for the i-th query
    # data_list[i][3]: Query execution plan of each configuration for the i-th query
    # data_list[i][4]: Details execution costs (each query is executed 4 times and the last 3 times are recorded) of each configuration for the i-th query

    data_list_string = read_csv(filepath)
    data, queries = [], []
    for i in range(len(data_list_string)):
        data_list_string[i][0] = ast.literal_eval(data_list_string[i][0])
        query = Query(data_list_string[i][0][0], data_list_string[i][0][1])
        query.columns = [column for column_name, column in column_dict.items() if column_name in query.text]
        queries.append(query)
        indexes_string = index_conversion(data_list_string[i][1])
        index_configurations = convert_configuration_to_obj(column_dict, indexes_string)
        average_costs = ast.literal_eval(data_list_string[i][2])
        plans = ast.literal_eval(data_list_string[i][3])
        data.append([query, index_configurations, average_costs, plans])
    return data, queries

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return F.mse_loss(input.view(-1), target.view(-1), reduction='mean')
    
class QLoss(nn.Module):
    def __init__(self, weight=None, min_val=1e-5, penalty_negative=1e5):
        self.min_val = min_val
        self.penalty_negative = penalty_negative
        super().__init__()

    def forward(self, input, target):
        qerror = []
        for i in range(len(target)):
            q_err = max(target[i]/input[i], input[i]/target[i])
            qerror.append(q_err)
        return torch.mean(torch.cat(qerror))

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.input_fc = nn.Linear(input_dim, 32)
        self.output_fc = nn.Linear(32, output_dim)
        
    def forward(self, x):
        h_1 = F.relu(self.input_fc(x))
        y_pred = self.output_fc(h_1)
        return y_pred
  
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.op_encoders = nn.ModuleList(nn.Linear(6,1) for _ in PHYSICAL_OPERATORS)
        self.fc = MLP(input_dim, output_dim)
        
    def forward(self, x):
        hidden = None
        for op, input in x["op_features"].items():
            id = PHYSICAL_OPERATORS.index(op)
            if hidden == None: hidden = self.op_encoders[id](input)
            else: hidden = torch.cat((hidden, self.op_encoders[id](input)), 1)

        hidden = torch.cat((F.leaky_relu(hidden), x["index_features"], x["table_stats"]),1)
        return self.fc(hidden)
      
class CustomDataset(Dataset):
    def __init__(self, operator_features, index_features, table_stats, labels):
        self.operator_features = np.array(operator_features)
        self.index_features = index_features
        self.table_stats = table_stats.to_numpy()
        self.labels = labels
        
    def __getitem__(self, index):
        return {"op_features": self.operator_features[index], "index_features": self.index_features[index], "table_stats": self.table_stats[index]}, self.labels[index]
    
    def __len__(self):
        return len(self.labels)
       
def has_child_node(query_plan):
    return "Plans" in query_plan.keys()

def has_filtering_property(query_plan):
    if "Filter" in query_plan.keys():
        return query_plan["Filter"]
    if "Hash Cond" in query_plan.keys():
        return query_plan["Hash Cond"]
    if "Join Filter" in query_plan.keys():
        return query_plan["Join Filter"]
    return ""

def is_join_operator(operator):
    return PHYISCAL_TO_LOGICAL_OPERATOR_MAP[operator] == "Join"

def is_sort_operator(operator):
    return PHYISCAL_TO_LOGICAL_OPERATOR_MAP[operator] == "Sort"

def is_aggregate_operator(operator):
    return PHYISCAL_TO_LOGICAL_OPERATOR_MAP[operator] == "Aggregate"

def is_scan_operator(operator):
    return PHYISCAL_TO_LOGICAL_OPERATOR_MAP[operator] == "Scan"

def get_table_from_plan_node(query_plan):
    table = ""
    if "Relation Name" in query_plan.keys():
        table = query_plan["Relation Name"]
    return table
        
def _extract_operator_feature(features, query_plan):
    if has_child_node(query_plan):
        for child_node in query_plan["Plans"]:
            _extract_operator_feature(features, child_node)
            
    current_operator = query_plan["Node Type"]
    op_feature = features[current_operator]
    op_feature["actual_rows"].append(query_plan["Actual Rows"])
    op_feature["actual_loops"].append(query_plan["Actual Loops"])
    op_feature["plan_rows"].append(query_plan["Plan Rows"])
    op_feature["plan_width"].append(query_plan["Plan Width"])
    op_feature["cost"].append(query_plan["Total Cost"] - query_plan["Startup Cost"])
    op_feature["actual_time"].append(query_plan["Actual Total Time"] - query_plan["Actual Startup Time"])

def extract_operator_features(query_plan):
    result = {op: {"actual_rows": [], "actual_loops": [], "plan_rows": [], "plan_width": [], "cost": [], "actual_time": []} for op in PHYSICAL_OPERATORS}
    _extract_operator_feature(result, query_plan)
    for op, v in result.items():
        result[op] = np.array([sum(v["actual_rows"])/len(v["actual_rows"]) if len(v["actual_rows"]) > 0 else 0, 
            sum(v["actual_loops"])/len(v["actual_loops"]) if len(v["actual_loops"]) > 0 else 0,
            sum(v["plan_rows"])/len(v["plan_rows"]) if len(v["plan_rows"]) > 0 else 0,
            sum(v["plan_width"])/len(v["plan_width"]) if len(v["plan_width"]) > 0 else 0,
            sum(v["cost"])/len(v["cost"]) if len(v["cost"]) > 0 else 0,
            sum(v["actual_time"])/len(v["actual_time"]) if len(v["actual_time"]) > 0 else 0,
        ])
    return result

def extract_shape_of_query_and_index(index, original_query_plan, indexed_query_plan):
    query_shape, index_shape = {}, []
    _extract_query_shape(query_shape, original_query_plan)
    visited = set()
    _extract_index_shape(index_shape, index, indexed_query_plan, visited)
    return query_shape, index_shape

def _extract_query_shape(query_shape, query_plan):
    current_operator = query_plan["Node Type"]
    logical_operator = PHYISCAL_TO_LOGICAL_OPERATOR_MAP[current_operator]
    if is_scan_operator(current_operator):
        table = get_table_from_plan_node(query_plan)
        if table in query_shape.keys():
            query_shape[table].append(logical_operator)
        else:
            query_shape[table] = [logical_operator]
        return table
    
    tables = []    
    if has_child_node(query_plan):
        for child_node in query_plan["Plans"]:
            table = _extract_query_shape(query_shape, child_node)
            if table and logical_operator:
                tables.append(table)
                query_shape[table].append(logical_operator)
    return tables[0] if 0<len(tables)<2 else ""

def _extract_index_shape(index_shape, index, query_plan, visited):
    current_operator = query_plan["Node Type"]
    logical_operator = PHYISCAL_TO_LOGICAL_OPERATOR_MAP[current_operator]
    if has_child_node(query_plan):
        for child_node in query_plan["Plans"]:
            _extract_index_shape(index_shape, index, child_node, visited)
            
    if (condition := has_filtering_property(query_plan)) != "":
        for column in index.columns:
            if column in visited: continue
            elif column.name in condition:
                index_shape.append(logical_operator)
                visited.add(column)
    elif is_sort_operator(current_operator) and "Sort Key" in query_plan.keys():
        sort_conditions = query_plan["Sort Key"]
        for sort_condition in sort_conditions:
            for column in index.columns:
                if column in visited: continue
                elif column.name in sort_condition:
                    index_shape.append(logical_operator)
                    visited.add(column)
    elif is_aggregate_operator(current_operator) and "Group Key" in query_plan.keys():
        aggregate_conditions = query_plan["Group Key"]
        for aggregate_condition in aggregate_conditions:
            for column in index.columns:
                if column in visited: continue
                elif column.name in aggregate_condition:
                    index_shape.append(logical_operator)
                    visited.add(column)
                    
def get_metadata_on_dataset(data):
    total_sample_count, max_config_len = 0, 0
    for entry in data:
        total_sample_count += len(entry[1])
        for index_config in entry[1]:
            max_config_len = max(max_config_len, len(index_config))
    return total_sample_count, max_config_len

# signal 3
def evaluate_operator_relevance(index, query_plan):
    result = {}
    _evaluate_operator_relevance(result, index, query_plan)
    return result

def _evaluate_operator_relevance(operator_relevance, index, query_plan):
    if has_child_node(query_plan):
        for child_node in query_plan["Plans"]:
            _evaluate_operator_relevance(operator_relevance, index, child_node)
            
    current_operator = query_plan["Node Type"]
    relevance = 0
    if (condition := has_filtering_property(query_plan)) != "":
        selectivities = [query_plan["Plan Rows"]/column.table.row_count for column in index.columns if column.name in condition]
        relevance = sum(selectivities)/len(selectivities) if len(selectivities) > 0 else 0
    elif is_sort_operator(current_operator) and "Sort Key" in query_plan.keys():
        densities = []
        conditions = query_plan["Sort Key"]
        for condition in conditions:
            for column in index.columns:
                if column.name in condition:
                    densities.append(column.cardinality/column.table.row_count)
        relevance = sum(densities)/len(densities) if len(densities) > 0 else 0
    elif is_aggregate_operator(current_operator) and "Group Key" in query_plan.keys():
        densities = []
        conditions = query_plan["Group Key"]
        for condition in conditions:
            for column in index.columns:
                if column.name in condition:
                    densities.append(column.cardinality/column.table.row_count)
        relevance = sum(densities)/len(densities) if len(densities) > 0 else 0
    if current_operator not in operator_relevance: 
        operator_relevance[current_operator] = []
    operator_relevance[current_operator].append(relevance)
    
def evaluate_configuration(column_dict, configuration, query_plan):
    configuration_features = {}
    _evaluate_configuration(column_dict, configuration_features, configuration, query_plan)
    return configuration_features

def _evaluate_configuration(column_dict, configuration_features, configuration, query_plan):
    current_operator = query_plan["Node Type"]
    if has_child_node(query_plan):
        for child_node in query_plan["Plans"]:
            _evaluate_configuration(column_dict, configuration_features, configuration, child_node)
    column_keys = [column_key for column_key,_ in column_dict.items()]          
    if (condition := has_filtering_property(query_plan)) != "":
        for index in configuration:
            for j, column in enumerate(index.columns):
                if column.name in condition:
                    configuration_features[f"selectivity_{column_keys.index(column.name)}"] = query_plan["Plan Rows"]/column.table.row_count
                    feature = f"operation_{column_keys.index(column.name)}"
                    if feature not in configuration_features.keys(): configuration_features[feature] = set()
                    configuration_features[feature].add(current_operator)
                    feature = f"position_{column_keys.index(column.name)}"
                    if  feature not in configuration_features.keys(): configuration_features[feature] = j
                    else: configuration_features[feature] = min(configuration_features[feature], j)
    elif is_aggregate_operator(current_operator) and "Group Key" in query_plan.keys():
        conditions = query_plan["Group Key"]
        for index in configuration:
            for j, column in enumerate(index.columns):
                for condition in conditions:
                    if column.name in condition:
                        feature = f"operation_{column_keys.index(column.name)}"
                        if feature not in configuration_features.keys(): configuration_features[feature] = set()
                        configuration_features[feature].add(current_operator)
                        feature = f"position_{column_keys.index(column.name)}"
                        if  feature not in configuration_features.keys(): configuration_features[feature] = j
                        else: configuration_features[feature] = min(configuration_features[feature], j)
    elif is_sort_operator(current_operator) and "Sort Key" in query_plan:
        conditions = query_plan["Sort Key"]
        for index in configuration:
            for j, column in enumerate(index.columns):
                for condition in conditions:
                    if column.name in condition:
                        feature = f"order_{column_keys.index(column.name)}"
                        if "DESC" in condition: configuration_features[feature] = "DESC"
                        else: configuration_features[feature] = "ASC" # ascending by default
                        feature = f"position_{column_keys.index(column.name)}"
                        if  feature not in configuration_features.keys(): configuration_features[feature] = j
                        else: configuration_features[feature] = min(configuration_features[feature], j)

def construct_table_features(data, table_dict):
    table_num = 25
    column_per_table = 34
    keys = [f"row_count_{i}" for i in range(table_num)]
    keys.extend([f"n_distinct_{i}_{j}" for i in range(table_num) for j in range(column_per_table)])
    keys.extend([f"null_frac_{i}_{j}" for i in range(table_num) for j in range(column_per_table)])
    table_stats = {k: 0 for k in keys}
    table_names = sorted(list(table_dict.keys()))
    table_list = list(table_dict[k] for k in table_names)
    for i, table in enumerate(table_list):
        table_stats[f"row_count_{i}"] = table.row_count
        for j, column in enumerate(table.columns):
            table_stats[f"n_distinct_{i}_{j}"] = column.cardinality
            table_stats[f"null_frac_{i}_{j}"] = column.null_frac
    sample_count, _ = get_metadata_on_dataset(data)
    for k, v in table_stats.items():
        table_stats[k] = [v for _ in range(sample_count)]
    table_stats = pd.DataFrame.from_dict(table_stats)
    return table_stats

def get_metadata_on_dataset(data):
    total_sample_count, max_config_len = 0, 0
    for entry in data:
        total_sample_count += len(entry[1])
        for index_config in entry[1]:
            max_config_len = max(max_config_len, len(index_config))
    return total_sample_count, max_config_len
                    
def construct_operator_index_features(data, table_dict, column_dict):
    sample_count, _ = get_metadata_on_dataset(data)
    index_feature_columns, labels = [], []
    operator_relevance_columns = [f"relevance_{operator}_of_index{i}" for i  in range(MAX_CONFIG_LEN) for operator in LOGICAL_OPERATORS]
    index_feature_columns.extend(operator_relevance_columns)
    query_shape_columns = [f"query_shape_operator{k}_on_table{j}_of_index{i}" for i in range(MAX_CONFIG_LEN) for j in range(MAX_TABLE_NUM+1) for k in range(5)]
    index_feature_columns.extend(query_shape_columns)
    index_shape_columns = [f"index_shape_operator{k}_of_index{i}" for i in range(MAX_CONFIG_LEN) for k in range(5)]
    index_feature_columns.extend(index_shape_columns)
    
    config_operation_columns = list(set(f"operation_{col}_{i}" for col in range(MAX_COLUMN_NUM) for i in range(3)))
    index_feature_columns.extend(config_operation_columns)
    config_position_columns = list(set(f"position_{col}" for col in range(MAX_COLUMN_NUM)))
    index_feature_columns.extend(config_position_columns)
    config_order_columns = list(set(f"order_{col}" for col in range(MAX_COLUMN_NUM)))
    index_feature_columns.extend(config_order_columns)
    
    operator_features = []
    index_features = pd.DataFrame(columns=index_feature_columns, index=range(sample_count))
    k = 0
    for entry in data:
        index_configs = entry[1]
        costs = entry[2]
        plans = entry[3]
        original_query_plan = plans[0] # no indexed query plan
        original_query_cost = costs[0] # no index query cost
        for j, index_config in enumerate(index_configs):
            labels.append(costs[j])
            indexed_query_plan = plans[j]
            for i, index in enumerate(index_config):
                query_shape, index_shape = extract_shape_of_query_and_index(index, original_query_plan, indexed_query_plan)
                table_keys = [table_key for table_key,_ in table_dict.items()]
                for table, operator_seq in query_shape.items():
                    for o, operator in enumerate(operator_seq):
                        if table in table_keys:
                            table_index = table_keys.index(table)
                            index_features.iloc[k][f"query_shape_operator{o}_on_{table_index}_of_index{i}"] = operator
                for o, operator in enumerate(index_shape):
                    index_features.iloc[k][f"index_shape_operator{o}_of_index{i}"] = operator
                relevance = evaluate_operator_relevance(index, original_query_plan)
                for operator in LOGICAL_OPERATORS:
                    if operator in relevance: index_features.iloc[k][f"relevance_{operator}_of_index{i}"] = sum(relevance[operator])/len(relevance[operator])
            
            config_feature = evaluate_configuration(column_dict, index_config, indexed_query_plan)
            for feature_name, value in config_feature.items():
                if isinstance(value, set):
                    for o, operator in enumerate(list(value)):
                        index_features.iloc[k][f"{feature_name}_{o}"] = operator
                else:
                    index_features.iloc[k][feature_name] = value
            operator_features.append(extract_operator_features(indexed_query_plan))
            k+=1

    index_features[operator_relevance_columns] = index_features[operator_relevance_columns].fillna(value=0)
    index_features[config_position_columns] = index_features[config_position_columns].fillna(value=len(config_position_columns))
    
    # onehot
    logical_operator_encoder = OneHotEncoder(sparse_output=False, categories=[LOGICAL_OPERATORS+[np.NaN] for _ in range(len(query_shape_columns)+len(index_shape_columns))])
    physical_operator_encoder = OneHotEncoder(sparse_output=False, categories=[PHYSICAL_OPERATORS+[np.NaN] for _ in range(len(config_operation_columns))])
    order_encoder = OneHotEncoder(sparse_output=False, categories=[["ASC", "DESC", np.NaN] for _ in range(len(config_order_columns))])
    features_num = index_features.select_dtypes(exclude="object")
    logical_operator_features_encoded = logical_operator_encoder.fit_transform(index_features[query_shape_columns + index_shape_columns])
    physical_operator_features_encoded = physical_operator_encoder.fit_transform(index_features[config_operation_columns])
    order_features_encoded = order_encoder.fit_transform(index_features[config_order_columns])
    index_features_encoded = np.concatenate((features_num.to_numpy(), logical_operator_features_encoded, physical_operator_features_encoded, order_features_encoded), axis=1)
    return index_features, index_features_encoded, operator_features, np.array(labels,dtype='float32')

def create_dataset(operator_features, index_features, table_stats, labels):
    dataset = CustomDataset(operator_features, index_features, table_stats, labels)
    batch_size = 64
    validation_split = .3
    shuffle_dataset = True
    random_seed = 2024

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    return train_loader, validation_loader



def train(model, optimizer, criterion, train_loader):
    print("training models...")
    model.train()
    path = "saved_model_stat_dict_copy.pt"
                        
    num_epochs = 10000
    for epoch in range(num_epochs):
        losses = []
        for batch_num, (batch_input, batch_label) in enumerate(train_loader):
            optimizer.zero_grad()
            for op, v in batch_input["op_features"].items():
                batch_input["op_features"][op] = v.to(device, torch.float32)
            batch_input["index_features"] = batch_input["index_features"].to(device, torch.float32)
            batch_input["table_stats"] = batch_input["table_stats"].to(device, torch.float32)
            
            batch_label = batch_label.to(device, torch.float32)
            output = model(batch_input)
            loss = criterion(output, batch_label)
            
            loss.backward()
            losses.append(loss.item())

            optimizer.step()

            if batch_num % 40 == 0:
                print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
        torch.save(model.state_dict(), path)
        print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))
        
        
        
def evaluation(model, validation_loader):
    model.eval()
    for _, (batch_input, batch_label) in enumerate(validation_loader):
        for op, v in batch_input["op_features"].items():
            batch_input["op_features"][op] = v.to(device, torch.float32)
        batch_input["index_features"] = batch_input["index_features"].to(device, torch.float32)
        batch_input["table_stats"] = batch_input["table_stats"].to(device, torch.float32)
        output = model(batch_input).cpu().detach().numpy().reshape(-1)
        batch_label = batch_label.numpy()
        print(q_error(batch_label, output))
        

if __name__ == "__main__":
    print("reading tables and data...")
    print("\treading tpc-ds 50...")
    table_dict_DS_50G, column_dict_DS_50G = read_table_info("../data/TPC_DS_50G/tpcds50trow.csv", "../data/TPC_DS_50G/tpcds50stats.csv")
    print("\treading tpc-ds 10...")
    table_dict_DS_10G, column_dict_DS_10G = read_table_info("../data/TPC_DS_10G/tpcds10trow.csv", "../data/TPC_DS_10G/tpcds10stats.csv")
    print("\treading tpc-h...")
    table_dict_H, column_dict_H = read_table_info("../data/TPCH/tpchtrow.csv", "../data/TPCH/tpchstats.csv")
    print("\treading imdb...")
    table_dict_IMDB, column_dict_IMDB = read_table_info("../data/IMDB/imdb_trows.csv", "../data/IMDB/imdb_stats.csv")
    print("\treading tpc-ds 50 queries...")
    DS_50G_data, _ = read_query_and_index_data("../data/TPC_DS_50G/TPC_DS_50GB.csv", column_dict_DS_50G)
    print("\treading tpc-ds 10 queries...")
    DS_10G_data, _ = read_query_and_index_data("../data/TPC_DS_10G/TPC_DS_10GB.csv", column_dict_DS_10G)
    print("\treading tpc-h queries...")
    H_data, _ = read_query_and_index_data("../data/TPCH/TPC_H_10.csv", column_dict_H)
    print("\treading imdb queries...")
    IMDB_data, _ = read_query_and_index_data("../data/IMDB/imdb_job.csv",column_dict_IMDB)
    print("generating index and operator features....")
    print("\tgenerating index and operator features for tpc-ds 50....")
    index_features_DS_50G, index_features_encoded_DS_50G, operator_features_DS_50G, labels_DS_50G = construct_operator_index_features(DS_50G_data, table_dict_DS_50G, column_dict_DS_50G)
    print("\tgenerating index and operator features for tpc-ds 10....")
    index_features_DS_10G, index_features_encoded_DS_10G, operator_features_DS_10G, labels_DS_10G = construct_operator_index_features(DS_10G_data, table_dict_DS_10G, column_dict_DS_10G)
    print("\tgenerating index and operator features for tpc-h....")
    index_features_H, index_features_encoded_H, operator_features_H, labels_H = construct_operator_index_features(H_data, table_dict_H, column_dict_H)
    print("\tgenerating index and operator features for imdb job....")
    index_features_JOB, index_features_encoded_JOB, operator_features_JOB, labels_JOB = construct_operator_index_features(IMDB_data, table_dict_IMDB, column_dict_IMDB)
    print("generating table statistics...")
    print("\tgenerating table statistics for tpc-ds 50...")
    table_stats_DS_50G = construct_table_features(DS_50G_data, table_dict_DS_50G)
    print("\tgenerating table statistics for tpc-ds 10...")
    table_stats_DS_10G = construct_table_features(DS_10G_data, table_dict_DS_10G)
    print("\tgenerating table statistics for tpc-h...")
    table_stats_H = construct_table_features(H_data, table_dict_H)
    print("\tgenerating table statistics for imdb job...")
    table_stats_JOB = construct_table_features(IMDB_data, table_dict_IMDB)
    
    # concate all datasets
    index_features = np.concatenate((index_features_encoded_DS_50G, index_features_encoded_DS_10G, index_features_encoded_H, index_features_encoded_JOB), axis=0)
    table_stats = pd.concat([table_stats_DS_50G, table_stats_DS_10G, table_stats_H, table_stats_JOB], axis=0)
    operator_features = operator_features_DS_50G + operator_features_DS_10G + operator_features_H + operator_features_JOB
    labels = np.concatenate((labels_DS_50G,labels_DS_10G,labels_H,labels_JOB), axis=0)
    
    print("preparing loaders...")
    train_loader, validation_loader = create_dataset(operator_features, index_features, table_stats, labels)
    
    INPUT_DIM = len(PHYSICAL_OPERATORS) + len(index_features_encoded_DS_50G[0]) + len(table_stats_DS_50G.columns)
    OUTPUT_DIM = 1

    # model = LinearModel(INPUT_DIM, OUTPUT_DIM).to(device)
    print("loading model...")
    # model = torch.load("saved_model.pt")
    model = LinearModel(INPUT_DIM, OUTPUT_DIM).to(device)
    model.load_state_dict(torch.load("saved_model_stat_dict_copy.pt"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0000005)
    criterion = MSELoss()
    print(model)
    train(model, optimizer, criterion, train_loader)