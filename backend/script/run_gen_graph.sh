# This script was written and developed by ABKGroup students at UCSD. 
# However, the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help 
# promote and foster the next generation of innovators.
#!/bin/bash
module unload genus
module load genus/21.1

mkdir log -p
export READ_SDC=1
genus -overwrite -log log/genus.log -no_gui -files run_genus_hybrid.tcl
mv syn_handoff syn_handoff_sdc

if [ -n "${RUN_GENERIC}" ]; then
  export READ_SDC=0
  if [ ${RUN_GENERIC} -eq 1 ]; then
    genus -overwrite -log log/genus.log -no_gui -files run_genus_hybrid.tcl
  fi
fi