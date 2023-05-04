#!/bin/tcsh
set base_dir="$argv[1]"
set design="$argv[2]"
set input_bit_width="$argv[3]"
set bit_width="$argv[4]"
set num_cycle="$argv[5]"
set log_num_cycle="$argv[6]"
set size="$argv[7]"
set num_unit="$argv[8]"

set script_dir="/home/fetzfs_projects/RTML/sakundu/Code/RTML/backend/script"

set prefix="${design}_${size}_${num_cycle}_${input_bit_width}_${num_unit}"
set run_dir="${base_dir}/${prefix}"
mkdir -p $run_dir
cd $run_dir

cp ${script_dir}/* .

#sed -i "s@xx_cp_xx@${tcp}@" accelerator.sdc
sed -i "s@xx_ip_bitwidth_xx@${input_bit_width}@" config.vh
sed -i "s@xx_bitwidth_xx@${bit_width}@" config.vh
sed -i "s@xx_num_cycle_xx@${num_cycle}@" config.vh
sed -i "s@xx_log_num_cycle_xx@${log_num_cycle}@" config.vh
sed -i "s@xx_size_xx@${size}@" config.vh
sed -i "s@xx_unit_xx@${num_unit}@" config.vh

set sign='`'
echo "${sign}define ${design} 1" >> config.vh

./run_gen_graph.sh

cp syn_handoff/*generic*.v ../accelerator_${prefix}.v
