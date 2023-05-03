#!/bin/python
'''
Input: Provide the run dir full path
'''
import os
import sys
import re

run_dir = sys.argv[1]
TOP_MODULE = 'accelerator'
util = os.getenv('UTIL')

base_name = os.path.basename(run_dir)
base_dir = os.path.dirname(run_dir)
run_type = os.path.basename(base_dir)
design_details = base_name.split('_')
benchmark = design_details[0]

if len(design_details) == 6:
    size = int(design_details[1])
    num_cycle = int(design_details[2])
    num_unit = int(design_details[3])
    bit_width = int(design_details[4])
    tcp = int(design_details[5])
    tcf = round(1000/tcp,6)
else:
    print(f"Run dir:{run_dir} is not valid")

file_summary = run_dir + "/rpt/invs_summary.rpt"
file_sdc = run_dir + "/rpt/" + TOP_MODULE + "_updated.sdc"
file_power = run_dir + "/rpt/invs_ff_power_updated.rpt"
log_file = run_dir + "log/run_innovus.log"

dc_power_rpt_file  = f"{run_dir}/rpt/{TOP_MODULE}_power_updated.rep"
dc_qor_rpt_file = f"{run_dir}/rpt/{TOP_MODULE}_qor_updated.rep"

flag = os.path.exists(file_summary)
flag = flag and os.path.exists(file_power)
flag = flag and os.path.exists(file_sdc)

num_macro = -1
num_std_cell = -1
std_cell_area = -1
macro_area = -1
core_area = -1

internal_power = -1
switching_power = -1
leakage_power = -1
total_power = -1
drc_count = -1

if flag:
    with open(file_summary) as f:
        contents = f.read().splitlines()
    f.close()

    # Design Details
    for line in contents:
        items = line.split()
        if (len(items) > 3 and items[0] == "#" and items[1] == "Hard" and 
            items[2] == "Macros:"):
            num_macro = int(items[-1])
        elif (len(items) > 3 and items[0] == "#" and items[1] == "Hard" and 
            items[2] == "Cells:"):
            num_std_cell = int(items[-1])
        elif (len(items) > 5 and items[0] == "Total" and items[1] == "area" and 
            items[2] == "of" and items[3] == "Standard" and 
            items[4] == "cells:"):
            std_cell_area = round(float(items[-2]),2)
        elif (len(items) > 4 and items[0] == "Total" and items[1] == "area" and 
            items[2] == "of" and items[3] == "Macros:"):
            macro_area = round(float(items[-2]),2)
        elif (len(items) > 4 and items[0] == "Total" and items[1] == "area" and 
            items[2] == "of" and items[3] == "Core:"):
            core_area = round(float(items[-2]),2)
        else:
            pass

    #Power
    with open(file_power) as f:
        contents = f.read().splitlines()
    f.close()

    for line in contents:
        items = line.split()
        if(len(items) > 2 and items[0] == "Total" and items[1] == "Internal"):
            internal_power = round(float(items[-2]), 2)
        elif(len(items) > 2 and items[0] == "Total" and items[1] == "Switching"):
            switching_power = round(float(items[-2]), 2)
        elif(len(items) > 2 and items[0] == "Total" and items[1] == "Leakage"):
            leakage_power = round(float(items[-2]), 2)
        elif(len(items) > 2 and items[0] == "Total" and items[1] == "Power:"):
            total_power = round(float(items[-1]), 2)
        else:
            pass


    # Timing
    with open(file_sdc) as f:
        contents = f.read().splitlines()
    f.close()

    items = contents[7].split()
    effective_clock_period = float(items[-1])
    effective_clock_frequency = round(1000 / effective_clock_period, 6)

if os.path.exists(log_file):
    with open(log_file) as f:
        contents = f.read().splitlines()
    f.close()

    for line in contents:
        items = line.split()
        if (len(items) > 6 and items[0] == "#Total" and items[1] == "number" and
            items[2] == "of" and items[3] == "DRC" and items[4] == "Violations"):
            drc_count = int(items[-1])
        else:
            pass

umap = {
    'mW':1e-3,
    'uW':1e-6,
    'nW':1e-9,
    'pW':1e-12,
    'W':1
}

flag = True
flag = flag and os.path.exists(dc_power_rpt_file)
flag = flag and os.path.exists(dc_qor_rpt_file)

if flag:
    with open(dc_power_rpt_file) as f:
        content = f.read()

    pattern = r'^Total\s+(\S+)'
    dc_internal_power = re.findall(pattern, content, re.M)
    dc_internal_power = dc_internal_power[-1]
    pattern = r'^Total\s+\S+\s+(\S+)'
    dc_internal_power_unit = re.findall(pattern, content, re.M)
    dc_internal_power_unit = dc_internal_power_unit[-1]
    dc_internal_power = float(dc_internal_power)*umap[dc_internal_power_unit]/umap['mW']

    pattern = r'^Total\s+\S+\s+\S+\s+(\S+)'
    dc_switching_power = re.findall(pattern, content, re.M)
    pattern = r'^Total\s+\S+\s+\S+\s+\S+\s+(\S+)'
    dc_switching_power_unit = re.findall(pattern, content, re.M)
    dc_switching_power_unit = dc_switching_power_unit[-1]
    dc_switching_power = float(dc_switching_power[-1])*umap[dc_switching_power_unit]/umap['mW']

    pattern = r'Cell Leakage Power\s+=\s+(\S+)'
    dc_leakage_power = re.findall(pattern, content, re.M)
    dc_leakage_power = dc_leakage_power[-1]
    pattern = r'Cell Leakage Power\s+=\s+\S+\s+(\S+)'
    dc_leakage_power_unit = re.findall(pattern, content, re.M)
    dc_leakage_power_unit = dc_leakage_power_unit[-1]
    dc_leakage_power = float(dc_leakage_power)*umap[dc_leakage_power_unit]/umap['mW']

    pattern = r'^Total\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)'
    dc_total_power = re.findall(pattern, content, re.M)
    pattern = r'^Total\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)'
    dc_total_power_unit = re.findall(pattern, content, re.M)
    dc_total_power_unit = dc_total_power_unit[-1]
    dc_total_power = float(dc_total_power[-1])*umap[dc_total_power_unit]/umap['mW']

    f.close()
    with open(dc_qor_rpt_file) as f:
        content = f.read()

    pattern = r'Critical Path Clk Period:\s+(\S+)'
    dc_effective_cp = re.findall(pattern, content, re.M)
    dc_effective_cp = dc_effective_cp[-1]
    dc_effective_cf = 1e3/float(dc_effective_cp)
    f.close()

else:
    dc_effective_cp = -1
    dc_internal_power = -1
    dc_leakage_power = -1
    dc_switching_power = -1
    dc_total_power = -1


print(f"{benchmark},{run_type},{size},{num_cycle},{num_unit},{bit_width},{util},{tcp},{tcf},"
        f"{drc_count},{num_macro},{num_std_cell},{macro_area},{std_cell_area},{core_area},"
        f"{switching_power},{leakage_power},{internal_power},{total_power},"
        f"{effective_clock_period},{effective_clock_frequency},"
        f"{dc_effective_cp},{dc_internal_power},{dc_switching_power},"
        f"{dc_leakage_power},{dc_total_power},{dc_effective_cf}"
        )
