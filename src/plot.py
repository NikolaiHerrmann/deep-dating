from deep_dating.summary import pipeline_compare
from deep_dating.util import read_serialize


if __name__ == "__main__":
    p = read_serialize("runs_v2/graphs/pipeline_results.pkl")
    pipeline_compare(*p)