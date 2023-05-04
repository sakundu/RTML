'''
Input: Provide the job file and RPT file
'''
from math import floor, log
from typing import List 
import sys
import os

def gen_gnu_job_list(job_file: str, run_dir: str, rpt_file: str,
                     num_cycles: List[int], sizes: List[int],
                     cp_util_list: List[List[int]], designs: List[str],
                     num_units: List[int], cp_step: float, 
                     run_file: str, ip_bits: List[int] = [4, 8]) -> None:
    
    # If run_dir does not exists create the run dir
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # Ensure the run_file exists
    if not os.path.exists(run_file):
        print(f'Please ensure the run script exists\nRun script: {run_file}')
        exit()
    
    if not os.path.isabs(run_dir):
        run_dir = os.path.abspath(run_dir)
    
    if not os.path.isabs(rpt_file):
        rpt_file = os.path.abspath(rpt_file)
    
    if len(num_units) == 1 and len(sizes) != 1:
        num_units = [num_units[0]]*len(sizes)
    
    # Write header of the report file #
    with open(rpt_file, "w") as f1:
        f1.write("benchmark,run_type,size,num_cycle,num_unit,bit_width,util,"
            "tcp,tcf,drc_count,num_macro,num_std_cell,macro_area,std_cell_area,"
            "core_area,switching_power,leakage_power,internal_power,"
            "total_power,effective_clock_period,effective_clock_frequency,"
            "dc_effective_cp,dc_internal_power,dc_switching_power,"
            "dc_leakage_power,dc_total_power,dc_effective_cf\n"
        )
    f2 = open(job_file, "w")
    
    # Generate the job file #
    for design in designs:
        for i in range(len(num_cycles)):
            num_cycle = num_cycles[i]
            log_cycle = int(floor(log(num_cycle)/log(2)) + 1)
            size = sizes[i]
            num_unit = num_units[i]
            for cp, util in cp_util_list:
                for ip_bit in ip_bits:
                    if cp_step != 0:
                        f2.write(f"tcsh {run_file} {design} {cp-cp_step} "
                            f"{ip_bit} {ip_bit*2} {num_cycle} {log_cycle} "
                            f"{size} {num_unit} {run_dir} {rpt_file} {util}\n")
                        f2.write(f"tcsh {run_file} {design} {cp} {ip_bit} "
                            f"{ip_bit*2} {num_cycle} {log_cycle} {size} "
                            f"{num_unit} {run_dir} {rpt_file} {util}\n")
                        f2.write(f"tcsh {run_file} {design} {cp+cp_step} "
                            f"{ip_bit} {ip_bit*2} {num_cycle} {log_cycle} "
                            f"{size} {num_unit} {run_dir} {rpt_file} {util}\n")
                    else:
                        f2.write(f"tcsh {run_file} {design} {cp} {ip_bit} "
                            f"{ip_bit*2} {num_cycle} {log_cycle} {size} "
                            f"{num_unit} {run_dir} {rpt_file} {util}\n")
    f2.close()


if __name__ == '__main__':
    job_file = sys.argv[1]
    run_dir = sys.argv[2]
    rpt_file = sys.argv[3]

    num_cycles = [4, 5, 8, 10, 11, 15, 20, 20, 5, 10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    sizes = [50, 40, 25, 20, 35, 25, 10, 45, 20, 50, 52, 31, 12, 41, 25, 19, 44, 10, 33, 16, 27, 50, 21, 37, 54, 29, 46, 8, 14, 35, 23, 48, 39, 6]
    cp_util_list = [[917, 0.408], [636, 0.592], [1098, 0.458], [787, 0.608], [826, 0.825], [869, 0.725], [689, 0.658], [613, 0.675], [662, 0.492], [552, 0.475], [970, 0.892], [1492, 0.558], [1818, 0.692], [1030, 0.758], [534, 0.542], [487, 0.708], [591, 0.525], [502, 0.775], [473, 0.642], [460, 0.575], [1265, 0.625], [571, 0.742], [751, 0.875], [719, 0.792], [518, 0.842], [1639, 0.808], [2325, 0.425], [2040, 0.858], [1176, 0.508], [1369, 0.442]]

    ip_bits = [4, 8]
    num_units = [1]

    designs = ['SVM']
    cp_step = 0

    run_file = '/home/fetzfs_projects/RTML/sakundu/Code/RTML/backend/job/'\
                'run_axiline_pnr.tcsh'
    
    gen_gnu_job_list(job_file, run_dir, rpt_file, num_cycles, sizes,
                     cp_util_list, designs, num_units, cp_step, run_file)
    