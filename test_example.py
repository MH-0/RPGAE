"""
this file is used to run the experiments on the dataset
"""

# loading libraries
import experiments.run_experiments as re

runs = 1

re.run("cora", 6, runs)
re.run("citeseer", 6, runs)
re.run("eumails", 6, runs)

re.run("usa_airtraffic", 3, runs)
re.run("europe_airtraffic", 3, runs)
re.run("brazil_airtraffic", 3, runs)

re.run("flydrosophilamedulla", 6, runs)
re.run("facebook", 6, runs)
re.run("socsignbitcoinalpha", 6, runs)
re.run("socsignbitcoinot", 6, runs)
re.run("ca-grqc", 4, runs)
