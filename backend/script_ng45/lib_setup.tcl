# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
# lib and lef, RC setup

set tmp_dir "/home/fetzfs_projects/RTML/sakundu/Code/RTML/backend/NanGate45"
set libdir "${tmp_dir}/lib"
set lefdir "${tmp_dir}/lef"
set qrcdir "${tmp_dir}/qrc"

set_db init_lib_search_path { \
    ${libdir} \
    ${lefdir} \
}

set libworst "  
    ${libdir}/NangateOpenCellLibrary_typical.lib \
    "

set libpower "
    ${libdir}/NangateOpenCellLibrary_typical.lib \
    "

set libbest " 
    ${libdir}/NangateOpenCellLibrary_typical.lib \
    "

set lef "
    ${lefdir}/NangateOpenCellLibrary.tech.lef \
    ${lefdir}/NangateOpenCellLibrary.macro.mod.lef \
    "

set qrc_max "${qrcdir}/NG45.tch"
set qrc_min "${qrcdir}/NG45.tch"
#
# Ensures proper and consistent library handling between Genus and Innovus
#set_db library_setup_ispatial true
setDesignMode -process 45
