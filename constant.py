TPC_DS_TABLE_PREFIX = {
    "dv": "dbgen_version",
    "ca": "customer_address",
    "cd": "customer_demographics",
    "d": "date_dim",
    "w": "warehouse",
    "sm": "ship_mode",
    "t": "time_dim",
    "r": "reason",
    "ib": "income_band",
    "i": "item",
    "s": "store",
    "cc": "call_center",
    "c": "customer",
    "web": "web_site",
    "sr": "store_returns",
    "hd": "household_demographics",
    "wp": "web_page",
    "p": "promotion",
    "cp": "catalog_page",
    "inv": "inventory",
    "cr": "catalog_returns",
    "wr": "web_returns",
    "ws": "web_sales",
    "cs": "catalog_sales",
    "ss": "store_sales"
}

TPC_H_TABLE_PREFIX = {
    "n": "nation",
    "r": "region",
    "p": "part",
    "s": "supplier",
    "ps": "partsupp",
    "c": "customer",
    "o": "orders",
    "l": "lineitem",
}

#
PHYISCAL_TO_LOGICAL_OPERATOR_MAP = {
    "Seq Scan": "Scan",
    "Bitmap Index Scan": "Scan",
    "Bitmap Heap Scan": "Scan",
    "Index Scan": "Scan",
    "Index Only Scan": "Scan",
    "CTE Scan": "Scan",
    "Subquery Scan": "Scan",
    "Sort": "Sort",
    "Incremental Sort": "Sort",
    "Hash Join": "Join",
    "Merge Join": "Join",
    "Nested Loop": "Join",
    "Aggregate": "Aggregate",
    "WindowAgg": "Aggregate",
    "Group": "Aggregate",
    "Gather Merge": "",
    "Gather": "",
    "BitmapOr": "",
    "BitmapAnd": "",
    "Limit": "",
    "Hash": "",
    "Result": "",
    "SetOp": "",
    "Append": "",
    "Materialize": "",
    "Unique": "",
    "Merge Append": "",
    "Memoize": "",
}


LOGICAL_OPERATORS = ["Scan", "Join", "Aggregate", "Sort"]
PHYSICAL_OPERATORS = list(PHYISCAL_TO_LOGICAL_OPERATOR_MAP.keys())