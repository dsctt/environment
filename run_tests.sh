#pytest -rP --benchmark-columns=ops,mean,stddev,min,max,iterations,rounds --benchmark-max-time=5 --benchmark-min-rounds=1 tests/test_performance.py
pytest --benchmark-columns=ops,mean,stddev,min,max,iterations,rounds --benchmark-max-time=5 --benchmark-min-rounds=1 tests/test_emulation.py
#pytest --benchmark-columns=ops,mean,stddev,min,max,iterations,rounds --benchmark-max-time=5 --benchmark-min-rounds=1
