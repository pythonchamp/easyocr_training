import time
from functools import wraps

class PerfCounter:
    @staticmethod
    def calculate_execution_time(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time for {func.__name__}: {execution_time} seconds")
            return result

        return wrapper