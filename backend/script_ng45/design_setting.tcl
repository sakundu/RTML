setMultiCpuUsage -localCpu 4
set design accelerator
set netlist "./gate/${design}.v"
set sdc "./gate/${design}.sdc"
set libdir "/home/fetzfs_projects/RTML/sakundu/Code/RTML/backend/NanGate45"

set lef "
    ${libdir}/lef/NangateOpenCellLibrary.tech.lef \
    ${libdir}/lef/NangateOpenCellLibrary.macro.mod.lef \
    "
    
set libworst "
    ${libdir}/lib/NangateOpenCellLibrary_typical.lib \
    "

set libpower "
    ${libdir}/lib/NangateOpenCellLibrary_typical.lib \
    "

set libbest "
    ${libdir}/lib/NangateOpenCellLibrary_typical.lib \
    "

set qrc_max "${libdir}/qrc/NG45.tch"
set qrc_min "${libdir}/qrc/NG45.tch"
 
#set layer_map "/home/zf4_techdata/libraries/GF_12nm/pdk/12LP/V1.0_2.1/PlaceRoute/Innovus/Techfiles/13M_3Mx_2Cx_4Kx_2Hx_2Gx_LB/12LP_13M_3Mx_2Cx_4Kx_2Hx_2Gx_LB_TQRC.map"
 
set site "FreePDK45_38x28_10R_NP_162NW_34O" 
 
set rptDir rpt
set encDir enc
 
if {![file exists $rptDir]} {
    exec mkdir $rptDir
}

if {![file exists $encDir]} {
    exec mkdir $encDir
}

# Since the inconsistency of time units of standard cells and macros, we need to set up the timing unit
# setLibraryUnit -time 1ps  -cap 0.001pf
set_default_switching_activity -input_activity 0.1 -seq_activity 0.1

# default settings
set init_pwr_net { VDD }
set init_gnd_net { VSS }
set init_verilog "$netlist"
set init_design_netlisttype "Verilog"
set init_design_settop 1
set init_top_cell "$design"
set init_lef_file "$lef"


# MCMM setup
create_library_set -name WC_LIB -timing $libworst
create_library_set -name BC_LIB -timing $libbest
create_library_set -name POWER_LIB -timing $libpower 

create_rc_corner -name Cmax -qx_tech_file $qrc_max -T 25
create_rc_corner -name Cmin -qx_tech_file $qrc_min -T 25
create_rc_corner -name Cpower -qx_tech_file $qrc_min -T 25
 
create_delay_corner -name WC -library_set WC_LIB -rc_corner Cmax
create_delay_corner -name BC -library_set BC_LIB -rc_corner Cmin
create_delay_corner -name POWERC -library_set POWER_LIB -rc_corner Cpower
 
create_constraint_mode -name CON -sdc_file $sdc
 
create_analysis_view -name WC_VIEW -delay_corner WC -constraint_mode CON
create_analysis_view -name BC_VIEW -delay_corner BC -constraint_mode CON
create_analysis_view -name POWER_VIEW -delay_corner POWERC -constraint_mode CON
 
set init_design_uniquify 1

init_design -setup {WC_VIEW} -hold {BC_VIEW}
 
set_analysis_view -leakage POWER_VIEW -dynamic POWER_VIEW -setup WC_VIEW -hold BC_VIEW
#set_analysis_view -leakage WC_VIEW -dynamic WC_VIEW -setup WC_VIEW -hold BC_VIEW
 
set_interactive_constraint_modes {CON}
 
setDesignMode -process 45 -powerEffort high


set_propagated_clock [all_clocks]
set_clock_propagation propagated

# design report
setAnalysisMode -reset
setAnalysisMode -analysisType onChipVariation
setAnalysisMode -checkType setup
setAnalysisMode -honorClockDomains false

group_path -name regs_pre_floorplan -from [all_registers] -to [all_registers]
group_path -name ingrp_pre_floorplan -from [all_inputs] -to [all_registers]
group_path -name outgrp_pre_floorplan -from [all_registers] -to [all_outputs]
 
## Report Generation script