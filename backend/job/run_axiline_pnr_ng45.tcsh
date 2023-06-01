#!/bin/tcsh
setenv UTIL 0.7
set design="$argv[1]"
set tcp="$argv[2]"
set input_bit_width="$argv[3]"
set bit_width="$argv[4]"
set num_cycle="$argv[5]"
set log_num_cycle="$argv[6]"
set size="$argv[7]"
set num_unit="$argv[8]"
set run_dir="$argv[9]"
set rpt_file="$argv[10]"
if ($#argv >= 11) then
    setenv UTIL "$argv[11]"
endif

setenv RUN_GENERIC 0
if ($#argv >= 12) then
  setenv RUN_GENERIC "$argv[12]"
endif

set script_dir="/home/fetzfs_projects/RTML/sakundu/Code/RTML/backend/script_ng45"
set job_dir="/home/fetzfs_projects/RTML/sakundu/Code/RTML/backend/job"
set graph_dir=`echo "$rpt_file" | sed 's@\.[^\.]*$@@'`

## Create Run Area and Copy file and udpate the files ##
## basedir name $run_dir/<design>_<size>_<num_cycle>_<num_unit>_<bit_width>_<tcp>_<util>
set prefix="${design}_${size}_${num_cycle}_${num_unit}_${bit_width}"
set p_dir="${run_dir}/${prefix}_${tcp}_${UTIL}"
echo "Run Dir: $p_dir"

mkdir -p $p_dir $graph_dir

cd $p_dir
cp -rf ${script_dir}/* .

sed -i "s@xx_cp_xx@${tcp}@" accelerator.sdc
sed -i "s@xx_ip_bitwidth_xx@${input_bit_width}@" config.vh
sed -i "s@xx_bitwidth_xx@${bit_width}@" config.vh
sed -i "s@xx_num_cycle_xx@${num_cycle}@" config.vh
sed -i "s@xx_log_num_cycle_xx@${log_num_cycle}@" config.vh
sed -i "s@xx_size_xx@${size}@" config.vh
sed -i "s@xx_unit_xx@${num_unit}@" config.vh

set sign='`'
echo "${sign}define ${design} 1" >> config.vh

## Starting the SPNR Run ##
source run_spnr
bash run_gen_graph.sh

## Generate the Report ##
set PYTHON_CMD="/home/zf4_projects/RTML/sakundu/env/sk-py/bin/python3.9"
$PYTHON_CMD ${job_dir}/extract_pnr_metrics.py $p_dir >> ${rpt_file}

### Extract the graphs ###
module unload iverilog
module load iverilog/v10_3

## TCP Based Graph ##
set graph_tcp_generic_graph="${graph_dir}/${prefix}_${tcp}.graph"
set graph_generic_verilog=`ls ${p_dir}/syn_handoff_sdc/*generic.v`
if ($graph_generic_verilog != "") then
  $PYTHON_CMD ${job_dir}/gen_graph.py $graph_generic_verilog \
                      $graph_tcp_generic_graph
endif

## Without TCP Based Graph ##
if ($RUN_GENERIC == "1") then
  set graph_generic_graph="${graph_dir}/${prefix}.graph"
  set graph_generic_verilog=`ls ${p_dir}/syn_handoff/*generic.v`
  if ($graph_generic_verilog != "") then
    $PYTHON_CMD ${job_dir}/gen_graph.py $graph_generic_verilog \
                        $graph_generic_graph
  endif
endif

## Delete the Run AREA ##
sleep 20
# rm -rf $p_dir
