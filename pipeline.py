import importlib
import pipeline_v6
importlib.reload(pipeline_v6)
from pipeline_v6 import query_sap

result = query_sap("Top 5 customers by total invoiced value", verbose=True)
print(result)