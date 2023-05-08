import time
from datetime import datetime
import os
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
import h2o
import numpy as np
import pandas as pd

class raytune:
    def atutuneDesignSpaceEvaluator(self, energy_model, config_df):
        pid = os.getpid()
        h2o_server_name = f'RTML_{pid}'
        h2o.init(nthreads = 16, max_mem_size = "20G", name = h2o_server_name)
        model = h2o.load_model(energy_model)
        test = h2o.H2OFrame(config_df)
        prediction = model.predict(test)
        config_df['predicted_energy'] = prediction.as_data_frame()
        h2o.shutdown()
        return config_df['predicted_energy'][0]
    
    def autotuneObjective(self, config):
        config_df = pd.DataFrame([config])
        min_energy = self.atutuneDesignSpaceEvaluator(self.energy_model, \
            config_df)
        
        # Return min_energy
        tune.report(minimum = min_energy)
    
    def __init__(self, num_samples, log_dir):
        self.energy_model = '/home/sakundu/Algo/RTML/Axiline_11062022/Output/DL_LHS_24_12_energy(uJ)'
        self.num_jobs = 4
        self.config = {
            'size': tune.choice([x for x in range(10, 51)]),
            'num_cycle': tune.choice([x for x in range(5, 21)]),
            'target_clock_frequency(GHz)': tune.uniform(0.5, 2.0),
            'num_unit': 1,
            'bit_width': 16,
            'benchmark_no': 3,
        }
        self.algo = HyperOptSearch()
        # User-defined concurrent #runs
        self.algo = ConcurrencyLimiter(self.algo, max_concurrent= self.num_jobs)
        self.scheduler = AsyncHyperBandScheduler()
        self.num_samples = num_samples
        self.log_dir = log_dir
        
    def __call__(self):
        start = time.time()
        analysis = tune.run(
            self.autotuneObjective,
            metric = 'minimum',
            mode = 'min',
            search_alg = self.algo,
            scheduler = self.scheduler,
            num_samples = self.num_samples,
            config = self.config,
            local_dir = self.log_dir       
        )
        end = time.time()
        runtime = round(end - start, 3)
        min_energy = analysis.best_result.get('minimum')
        best_config = analysis.best_config
        num_cycle = best_config.get('num_cycle')
        size = best_config.get('size')
        tcf = best_config.get('target_clock_frequency(GHz)')
        print(f'Runtime: {runtime} min_energy: {min_energy}')
        print(f'num_cycle: {num_cycle}, size: {size}, tcf:{tcf}')
        return

if __name__ == '__main__':
    dir_pref = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'./log_{dir_pref}'
    rt_ob = raytune(100, log_dir)
    # rt_ob.config = {
    #         'size': 25,
    #         'num_cycle': 8,
    #         'target_clock_frequency(GHz)': 1.0,
    #         'num_unit': 1,
    #         'bit_width': 16,
    #         'benchmark_no': 3,
    #     }
    # print(rt_ob.autotuneObjective(rt_ob.config))
    rt_ob()