#!/home/shijiachen/anaconda3/envs/siyuan/bin/python
import json
import csv
import logging

from selection.index_selection_evaluation import DBMSYSTEMS
from selection.query_generator import QueryGenerator
from selection.table_generator import TableGenerator
from selection.what_if_index_creation import WhatIfIndexCreation
from selection.workload import Workload
from selection.candidate_generation import syntactically_relevant_indexes

        
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
    what_if = WhatIfIndexCreation(db_connector)
    query_generator = QueryGenerator(
        config["benchmark_name"],
        config["scale_factor"],
        db_connector,
        config["queries"],
        table_generator.columns,
    )
    workload = Workload(query_generator.queries)
    for table in table_generator.tables:
        row_count = db_connector.table_row_count(table.name)
        table.set_row_count(row_count)
        for column in table.columns:
            card = db_connector.get_column_cardinality(column)
            column.set_cardinality(-card * row_count if card < 0 else card)
    query_plans_with_index, query_costs_with_index, query_plans = {}, {}, []
    for query in workload.queries:
        logging.info(f"Now generating statistics for query {query.nr}.")
        original_query_plan = db_connector.get_plan_with_statistics(query)
        query_plans_with_index[(query, None)] = original_query_plan
        query_costs_with_index[(query, None)] = db_connector.get_cost(query)
        query_plans.append(original_query_plan)
        indexes = syntactically_relevant_indexes(query, 2)
        indexes = indexes[int(0.5*len(indexes))]
        for index in indexes:
            what_if.simulate_index(index)
            indexed_query_plan = db_connector.get_plan_with_statistics(query)
            indexed_query_cost = db_connector.get_cost(query)
            what_if.drop_simulated_index(index)
            query_plans_with_index[(query,index)] = indexed_query_plan
            query_costs_with_index[(query,index)] = indexed_query_cost
            # with open("../TPC-H.csv", "a+") as f:
            #     writer = csv.writer(f, delimiter=',')
            #     original_query_cost = query_costs_with_index[(query, None)]
            #     original_query_plan = query_plans_with_index[(query, None)]
            #     data = [str([query.nr,query.text]), str([[], index]), str([original_query_cost,query_costs_with_index[(query, index)]]), str([original_query_plan,indexed_query_plan])]
            #     writer.writerow(data)
            query_plans.append(indexed_query_plan)
        logging.info(f"Query {query.nr} statistics is completed.")
    return query_plans_with_index, query_costs_with_index, query_plans



if __name__ == '__main__':
    run()
