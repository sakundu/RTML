# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
source lib_setup.tcl
source design_setup.tcl

# set the output directories
set HANDOFF_PATH syn_handoff

if {![file exists ${HANDOFF_PATH}]} {
  file mkdir ${HANDOFF_PATH}
}

# set threads
set_db max_cpus_per_server 16
set_db super_thread_servers "localhost"

# Libraries
set list_lib "$libworst"


# Target library
set link_library $list_lib
set target_library $list_lib


set_db hdl_flatten_complex_port true
set_db hdl_record_naming_style %s_%s
set_db auto_ungroup none

set_db library $list_lib

# Exclude Scan flops
set_dont_use [get_lib_cells *sdf*]

#################################################
# Load Design and Initialize
#################################################
set_db init_hdl_search_path $rtldir 
source rtl_list.tcl

foreach rtl_file $rtl_all {
    read_hdl -sv $rtl_file
}

elaborate $DESIGN
time_info Elaboration

if {![info exist ::env(READ_SDC)] || $::env(READ_SDC) == 0} {
    source $sdc
}

init_design
check_design -unresolved
check_timing_intent

# keep hierarchy during synthesis
set_db auto_ungroup none
set_db minimize_uniquify true

syn_generic
write_hdl -generic > ${HANDOFF_PATH}/${DESIGN}_generic.v
time_info GENERIC

exit
