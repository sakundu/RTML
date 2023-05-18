import numpy as np
import pandas as pd
import sys

def updateAxilineDf(df, pref = None):
    l = df.shape[0]
    vector_axiline = {
        2: 500000,
        3: 500000,
        1: 500000,
        4: 500000
    }

    benchmarkMap = {'LINEAR':1, 'LOGISTIC':2, 'SVM':3, 'RECO':4}
    if pref == None:
        df['energy(uJ)'] = [0.0]*l
        df['runtime(ms)'] = [0.0]*l
        df['vector_count'] = [0]*l
        df['benchmark_no'] = [0]*l
    else:
        df[pref+'_energy(uJ)'] = [0.0]*l
        df[pref+'_runtime(ms)'] = [0.0]*l

    for i in range(l):
        design = df.loc[i,'benchmark']
        numCycle = df.loc[i,'num_cycle']
        clockCycle = findClockCycle(numCycle, 
                                    vector_axiline[benchmarkMap[design]])
        if pref == None:
            eff_cp = df.loc[i,'effective_clock_period(ps)']
            power = df.loc[i,'total_power(mW)']
            runtime = eff_cp*clockCycle*1e-9
            energy = power*runtime
            df.at[i, 'vector_count'] = vector_axiline[benchmarkMap[design]]
            df.at[i, 'energy(uJ)'] = energy
            df.at[i, 'runtime(ms)'] = runtime
            df.at[i, 'benchmark_no'] = benchmarkMap[design]
        else:
            eff_cp = df.loc[i,pref+'_effective_clock_period(ps)']
            power = df.loc[i,pref+'_total_power(mW)']
            runtime = eff_cp*clockCycle*1e-9
            energy = power*runtime
            df.at[i, pref+'_energy(uJ)'] = energy
            df.at[i, pref+'_runtime(ms)'] = runtime
    return df

def findClockCycle (numCycle, vectorCount):
    z = numCycle*2 + 1
    return z + (z - numCycle + 1)*(vectorCount - 1)

def axilineSpnrDataGen(pnr_df, dc_df):
    pnr_df.rename(columns={'tcp' : 'target_clock_period(ps)', 
                'tcf' : 'target_clock_frequency(GHz)',
                'effective_clock_period' : 'effective_clock_period(ps)',
                'effective_clock_frequency' : 'effective_clock_frequency(GHz)',
                'switching_power' : 'switching_power(mW)',
                'leakage_power' : 'leakage_power(mW)',
                'internal_power' : 'internal_power(mW)',
                'total_power' : 'total_power(mW)'
                }, inplace = True
            )
    if 'core_area' in pnr_df.columns:
        pnr_df.rename(columns={'core_area' : 'corea_area(um^2)'}, 
                      inplace = True)
        
        
    # dc_df = pd.read_csv(dc_rpt)
    dc_df.rename(
        columns={
            'tcp' : 'target_clock_period(ps)', 
            'tcf' : 'target_clock_frequency(GHz)',
            'dc_effective_cp' : 'dc_effective_clock_period(ps)',
            'dc_switching_power' : 'dc_switching_power(mW)',
            'dc_leakage_power' : 'dc_leakage_power(mW)',
            'dc_internal_power' : 'dc_internal_power(mW)',
            'dc_total_power' : 'dc_total_power(mW)'
            }, inplace= True
        )
    dc_df['dc_effective_clock_frequency(GHz)'] = \
            1e3/dc_df['dc_effective_clock_period(ps)']
    dc_df = dc_df.drop(['target_clock_frequency(GHz)'], axis = 1)
    dc_df = updateAxilineDf(dc_df, pref = 'dc')
    pnr_df = updateAxilineDf(pnr_df)
    spnr_df = pnr_df.merge(dc_df, how = 'inner', on = ['benchmark', 'run_type', 
                        'size', 'num_cycle', 'num_unit', 'bit_width', 
                        'target_clock_period(ps)'])
    
    return spnr_df

def main_prep_data(spnr_rpt: str, train_rpt: str) -> None:
    spnr_df = pd.read_csv(spnr_rpt)
    config_columns = ['benchmark', 'run_type', 'size', 'num_cycle', 'num_unit',
                        'bit_width', 'tcp', 'util']

    spnr_df = spnr_df.drop_duplicates(subset=config_columns).reset_index(drop=True)

    pnr_columns = ['benchmark', 'run_type', 'size', 'util', 'core_area', 
                'std_cell_area', 'num_cycle', 'num_unit',
                'bit_width', 'tcp', 'tcf', 'switching_power', 'leakage_power',
                'internal_power', 'total_power', 'effective_clock_period',
                'effective_clock_frequency']

    dc_columns = ['benchmark', 'run_type', 'size', 'num_cycle', 'num_unit', 
            'tcp', 'bit_width', 'tcf','dc_effective_cp', 'dc_internal_power',
            'dc_switching_power', 'dc_leakage_power', 'dc_total_power']

    pnr_df = spnr_df[pnr_columns].copy().reset_index(drop=True)
    dc_df = spnr_df[dc_columns].copy().reset_index(drop=True)

    updated_spnr_df = axilineSpnrDataGen(pnr_df, dc_df)
    updated_spnr_df.to_csv(train_rpt, index=False)
    

if __name__ == '__main__':
    spnr_rpt = sys.argv[1]
    train_rpt = sys.argv[2]

    main_prep_data(spnr_rpt, train_rpt)