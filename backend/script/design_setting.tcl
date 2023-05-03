setMultiCpuUsage -localCpu 4
set design accelerator
set netlist "./gate/${design}.v"
set sdc "./gate/${design}.sdc"
set libdir "/home/zf4_projects/RTML/sakundu/PNR/REF/gf12/"

set lef "
    ${libdir}/lef/12LP_13M_3Mx_2Cx_4Kx_2Hx_2Gx_LB_84cpp_tech.lef \
    ${libdir}/lef/sc9mcpp84_12lp_base_lvt_c14.lef \
    ${libdir}/lef/sc9mcpp84_12lp_base_rvt_c14.lef \
    ${libdir}/lef/sc9mcpp84_12lp_base_slvt_c14.lef \
    "
    
set libworst "
    ${libdir}/lib/sc9mcpp84_12lp_base_lvt_c14_sspg_sigcmax_max_0p72v_125c.lib \
    ${libdir}/lib/sc9mcpp84_12lp_base_rvt_c14_sspg_sigcmax_max_0p72v_125c.lib \
    ${libdir}/lib/sc9mcpp84_12lp_base_slvt_c14_sspg_sigcmax_max_0p72v_125c.lib \
    "


set libpower "
    ${libdir}/lib/sc9mcpp84_12lp_base_lvt_c14_ffpg_sigcmin_min_0p88v_125c.lib \
    ${libdir}/lib/sc9mcpp84_12lp_base_rvt_c14_ffpg_sigcmin_min_0p88v_125c.lib \
    ${libdir}/lib/sc9mcpp84_12lp_base_slvt_c14_ffpg_sigcmin_min_0p88v_125c.lib \
    "

set libbest "
    ${libdir}/lib/sc9mcpp84_12lp_base_lvt_c14_ffpg_sigcmin_min_0p88v_m40c.lib \
    ${libdir}/lib/sc9mcpp84_12lp_base_rvt_c14_ffpg_sigcmin_min_0p88v_m40c.lib \
    ${libdir}/lib/sc9mcpp84_12lp_base_slvt_c14_ffpg_sigcmin_min_0p88v_m40c.lib \
    "


set qrc_max "${libdir}/qrc/SigCmax/qrcTechFile"
set qrc_min "${libdir}/qrc/SigCmin/qrcTechFile"
 
#set layer_map "/home/zf4_techdata/libraries/GF_12nm/pdk/12LP/V1.0_2.1/PlaceRoute/Innovus/Techfiles/13M_3Mx_2Cx_4Kx_2Hx_2Gx_LB/12LP_13M_3Mx_2Cx_4Kx_2Hx_2Gx_LB_TQRC.map"
 
set site "sc9mcpp84_12lp" 
 
set rptDir rpt
set encDir enc
 
if {![file exists $rptDir]} {
    exec mkdir $rptDir
}

if {![file exists $encDir]} {
    exec mkdir $encDir
}

# Since the inconsistency of time units of standard cells and macros, we need to set up the timing unit
setLibraryUnit -time 1ps  -cap 0.001pf
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

create_rc_corner -name Cmax -qx_tech_file $qrc_max -T 125
create_rc_corner -name Cmin -qx_tech_file $qrc_min -T -40
create_rc_corner -name Cpower -qx_tech_file $qrc_min -T 125
 
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
 
setDesignMode -process 12 -powerEffort high


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
 
report_timing -path_group  regs_pre_floorplan -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/regs_pre_floorplan.mtarpt
report_timing -path_group  ingrp_pre_floorplan -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/ingrp_pre_floorplan.mtarpt
report_timing -path_group  outgrp_pre_floorplan -machine_readable -max_paths 10000 -max_slack 3000 > ${rptDir}/outgrp_pre_floorplan.mtarpt
 


summaryReport -noHtml -outfile ${rptDir}/pre_flooprlan_summary.rpt
report_power -outfile ${rptDir}/pre_floorplan_power.rpt
report_timing -path_type full_clock  > ${rptDir}/pre_floorplan_timing.rpt
report_power -leakage -outfile ${rptDir}/pre_floorplan_leakage_power.rpt
timeDesign -prePlace >  ${rptDir}/pre_floorplan_timeDesign.rpt



set lvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TL_C14].cell]
set num_lvt_cell [llength $lvt_cell_list]

set rvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TR_C14].cell]
set num_rvt_cell [llength $rvt_cell_list]

set slvt_cell_list [dbGet [dbGet -p2 top.insts.cell.name *TSL_C14].cell]
set num_slvt_cell [llength $slvt_cell_list]

set fp [open "${rptDir}/pre_floorplan_cell_distribution.rpt" w]

puts $fp "The number of slvt std cells:   $num_slvt_cell"
puts $fp "The number of lvt std cells:  $num_lvt_cell"
puts $fp "The number of rvt std cells:   $num_rvt_cell"


puts $fp "\n"
puts $fp "The details of slvt std cells:  "
foreach cell $slvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of lvt std cells:  "
foreach cell $lvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

puts $fp "\n"
puts $fp "The details of rvt std cells:  "
foreach cell $rvt_cell_list {
    set cell_name [dbGet $cell.name]
    puts $fp $cell_name
}

close $fp



defOut -routing ${encDir}/pre_floorplan_${design}.def
saveNetlist ${encDir}/pre_floorplan_${design}.v
saveDesign ${encDir}/pre_floorplan_${design}.enc



