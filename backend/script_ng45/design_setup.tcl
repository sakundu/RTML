# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.

set DESIGN accelerator
set rtldir ". /home/zf4_projects/RTML/sakundu/PNR/ICCAD22/RTL/inference"
set sdc  ./${DESIGN}.sdc

# Effort level during optimization in syn_generic -physical (or called generic) stage
# possible values are : high, medium or low
set GEN_EFF medium

# Effort level during optimization in syn_map -physical (or called mapping) stage
# possible values are : high, medium or low
set MAP_EFF high
#
set SITE "unithd"
set HALO_WIDTH 5
set TOP_ROUTING_LAYER 10
