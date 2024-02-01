#!/home/shijiachen/anaconda3/envs/siyuan/bin/python
import json
import csv
import logging
import itertools
import random
import os
import sys

from selection.index_selection_evaluation import DBMSYSTEMS
from selection.query_generator import QueryGenerator
from selection.table_generator import TableGenerator
from selection.what_if_index_creation import WhatIfIndexCreation
from selection.cost_evaluation import CostEvaluation
from selection.workload import Workload
from selection.index import Index, index_merge
from selection.candidate_generation import syntactically_relevant_indexes, candidates_per_query
from selection.utils import get_utilized_indexes, indexes_by_table


def sample_candidates(candidates_per_query, max_config_width):
    candidates = [[]]
    for width in range(1, max_config_width+1):
        possible_candidates = random.sample(candidates_per_query, width)
        if width == 1:
            candidates.append(possible_candidates)
            continue
        else:
            # check if a config contains same column index
            column_check = set()
            for candidate in possible_candidates:
                if column_check & set(candidate.columns): possible_candidates.remove(candidate)
                else: column_check |= set(candidate.columns)
            # keep sample until reaches the width
            while len(possible_candidates) < width:
                candidate = random.sample(candidates_per_query, 1)[0]
                if column_check & set(candidate.columns): continue
                else: 
                    column_check |= set(candidate.columns)
                    possible_candidates.append(candidate)
        candidates.append(possible_candidates)
    return candidates

def run():
    config_file = "config.json"
    with open(config_file) as f:
        config = json.load(f)
    dbms_class = DBMSYSTEMS[config["database_system"]]
    generating_connector = dbms_class(None, autocommit=True)
    table_generator = TableGenerator(config["benchmark_name"], config["scale_factor"], generating_connector)
    database_name = table_generator.database_name()
    database_system = config["database_system"]
    db_connector = DBMSYSTEMS[database_system](database_name)
    query_generator = QueryGenerator(
        config["benchmark_name"],
        config["scale_factor"],
        db_connector,
        config["queries"],
        table_generator.columns,
    )
    workload = Workload(query_generator.queries)
    cost_evaluation = CostEvaluation(db_connector, cost_estimation="actual_runtimes")
    # Generate syntactically relevant candidates
    candidates = candidates_per_query(workload,2,candidate_generator=syntactically_relevant_indexes)
    print(f"{len(candidates)} candidates are generated")
    number_of_actual_runs = 4

    filename = "../data/DSB/dsb.csv"
    iter = 48
    for query, candidate_per_query in zip(workload.queries[iter:], candidates[iter:]):
        print(f"iteration no: {iter}: ")
        entry = [[query.nr, query.text]]
        index_configs_per_query = sample_candidates(candidate_per_query, 4)
        formatted_index_configs_per_query = []
        average_execution_times_per_index_config, execution_time_list_per_config, query_plans_per_index_config = [], [], []
        for j, index_config in enumerate(index_configs_per_query):
            if len(index_config) == 0: formatted_index_configs_per_query.append([])
            elif len(index_config) == 1: formatted_index_configs_per_query.append(index_config[0])
            else: formatted_index_configs_per_query.append(tuple(index_config))
            cost_evaluation._prepare_cost_calculation(index_config)
            execution_time_list = []
            for i in range(number_of_actual_runs):
                print(f"\tnow running the {i}th run on {j}th index config")
                actual_execu_time, plan = db_connector.exec_query(query)
                execution_time_list.append(actual_execu_time)
            average_execution_time = sum(execution_time_list)/len(execution_time_list)
            average_execution_times_per_index_config.append(average_execution_time)
            execution_time_list_per_config.append(execution_time_list[1:])
            query_plans_per_index_config.append(plan)
            cost_evaluation.complete_cost_estimation()
        entry.append(formatted_index_configs_per_query)
        entry.append(average_execution_times_per_index_config)
        entry.append(query_plans_per_index_config)
        entry.append(execution_time_list_per_config)
        with open(filename, "a+") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(entry)
        iter+=1

if __name__ == '__main__':
    pid = os.getpid()
    print(f"Current running pid: {pid}")
    run()
