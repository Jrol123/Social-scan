from dotenv import dotenv_values
from pandas import read_csv
from src.get_report import form_report

if __name__ == "__main__":
    # https://stackoverflow.com/a/78749746
    secrets = dotenv_values()
    
    summaries = read_csv("examples/04_clusterization/clustered_summaries2.csv", index_col=0)
    clusters = read_csv("examples/04_clusterization/categories.csv", index_col=0)
    
    form_report(summaries, clusters, secrets["CHUTES_API_TOKEN"], "examples/report.pdf")
