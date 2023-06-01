###############################
source design_setting.tcl
source floorplan.tcl

source /home/fetzfs_projects/RTML/sakundu/Code/RTML/backend/NanGate45/util/pdn_config.tcl
source pdn_flow.tcl

# saveDesign $encDir/post_floorplan_${design}.enc
# saveNetlist $encDir/post_floorplan_${design}.v
# defOut -routing  $encDir/post_floorplan_${design}.def

###########################################
setFillerMode -fitGap true
# Specifies the minimum sites gap between instances
setPlaceMode -place_detail_legalization_inst_gap 1
setDesignMode -topRoutingLayer 10
setDesignMode -bottomRoutingLayer 2 
# Enables placer to honor and fix double pattern constaint violations between 
# adjacent cells
setPlaceMode -place_detail_color_aware_legal true
setPlaceMode -place_global_place_io_pins true
place_opt_design -out_dir $rptDir -prefix place
refinePlace

# design report
setAnalysisMode -reset
setAnalysisMode -analysisType onChipVariation
setAnalysisMode -checkType setup
setAnalysisMode -honorClockDomains false


##############################################
setOptMode -unfixClkInstForOpt false
create_ccopt_clock_tree_spec -file $design.ccopt
ccopt_design

# Use actual clock network
set_interactive_constraint_modes [all_constraint_modes -active]
set_propagated_clock [all_clocks]
set_clock_propagation propagated

# Post-CTS timing optimization
setOptMode -powerEffort high -leakageToDynamicRatio 0.5
setOptMode -usefulSkew true
optDesign -postCTS -hold

# design report
setAnalysisMode -reset
setAnalysisMode -analysisType onChipVariation
setAnalysisMode -checkType setup
setAnalysisMode -honorClockDomains false


##################################################################
setNanoRouteMode -routeTopRoutingLayer 10
setNanoRouteMode -routeBottomRoutingLayer 2
setNanoRouteMode -routeWithSiDriven true
setNanoRouteMode -routeWithTimingDriven true
setNanoRouteMode -routeExpUseAutoVia true

## Fix antenna violations
setNanoRouteMode -routeInsertAntennaDiode true
setNanoRouteMode -drouteFixAntenna true

## Recommended by lib owners
# Prevent router modifying M1 pins shapes
setNanoRouteMode -routeWithViaInPin "1:1"
setNanoRouteMode -routeWithViaOnlyForStandardCellPin "1:1"

## minimizes via count during the route
setNanoRouteMode -routeConcurrentMinimizeViaCountEffort high

## allows route of tie off nets to internal cell pin shapes rather than routing to special net structure.
setNanoRouteMode -routeAllowPowerGroundPin true

## weight multi cut use high and spend more time optimizing dcut use.
setNanoRouteMode -drouteUseMultiCutViaEffort high

## limit VIAs to ongrid only for VIA1 (S1)
setNanoRouteMode -drouteOnGridOnly "via 1:1"
setNanoRouteMode -drouteAutoStop false

#SM suggestion for solving long extraction runtime during GR
setNanoRouteMode -grouteExpWithTimingDriven false

routeDesign

# design report
setAnalysisMode -reset
setAnalysisMode -analysisType onChipVariation
setAnalysisMode -checkType setup
setAnalysisMode -honorClockDomains false

## Report

###########################################################################
# fix drc
verify_drc -limit 100000000
editDeleteViolations
routeDesign

verifyProcessAntenna -error 100000000 -maxFloatingAreaDiffNet -pgnet
verify_drc -limit 100000000

ecoRoute -fix_drc

fixVia -minStep
verify_drc -limit 100000000

fixVia -minStep
verify_drc -limit 100000000

# design report
setAnalysisMode -reset
setAnalysisMode -analysisType onChipVariation
setAnalysisMode -checkType setup
setAnalysisMode -honorClockDomains false

# Report

############################################################################
## post route optimization
setDelayCalMode -reset
setDelayCalMode -SIAware true
setExtractRCMode -engine postRoute -coupled true -effortLevel medium
setAnalysisMode -reset
setAnalysisMode -honorClockDomains false
setAnalysisMode -analysisType onChipVariation -cppr both
setOptMode -powerEffort high -leakageToDynamicRatio 0.5

# Report

optDesign -postRoute -hold -setup

# Report

# leakage recovery
setOptMode -leakageToDynamicRatio 0.5
optPower -postRoute  -effortLevel high

##################################################
# Report Design
# SPEF generation
setDelayCalMode -reset
setDelayCalMode -SIAware true
setExtractRCMode -engine postRoute -coupled true -effortLevel medium
setAnalysisMode -reset
setAnalysisMode -honorClockDomains false
setAnalysisMode -analysisType onChipVariation -cppr both


extractRC
 
 
# rcOut -rc_corner Cmax -spef ${encDir}/invs_Cmax_$design\.spef
# rcOut -rc_corner Cmin -spef ${encDir}/invs_Cmin_$design\.spef


# Report

summaryReport -noHtml -outfile ${rptDir}/invs_summary.rpt
report_power -outfile ${rptDir}/invs_power.rpt
report_timing -path_type full_clock > ${rptDir}/invs_timing.rpt
report_power -leakage -outfile ${rptDir}/invs_leakage_power.rpt
report_power -view POWER_VIEW -outfile ${rptDir}/invs_ff_power.rpt
report_power -view WC_VIEW -outfile ${rptDir}/invs_ss_power.rpt
timeDesign -postRoute >  ${rptDir}/invs_timeDesign.rpt
write_sdc > ${rptDir}/invs_route.sdc

###############################################################
#### Write Updated Report
set path_collection [report_timing -collection]
set WNS 0
 
foreach_in_collection path $path_collection {
    set WNS [get_property $path slack]
}
 
set clock_period 0 
set clock_periods [get_property [get_clocks] period]
set clock_period [lindex $clock_periods 0]

puts "Clock Period: $clock_period, WNS: $WNS"
set effective_clock_period [expr $clock_period - $WNS]
 
write_sdc > ${rptDir}/${design}_updated.sdc
 
exec sed -i "s/period/period@/g" ${rptDir}/${design}_updated.sdc
exec sed -i "s/@.*//g" ${rptDir}/${design}_updated.sdc
exec sed -i "s/period/period  ${effective_clock_period}/g" ${rptDir}/${design}_updated.sdc
 
update_constraint_mode -ilm_sdc_files $sdc -name CON -sdc_files ${rptDir}/${design}_updated.sdc
set_propagated_clock [all_clocks]
set_clock_propagation propagated 
 
setDelayCalMode -reset
setDelayCalMode -SIAware true
setExtractRCMode -engine postRoute -coupled true -effortLevel medium
setAnalysisMode -reset
setAnalysisMode -honorClockDomains false
setAnalysisMode -analysisType onChipVariation -cppr both

summaryReport -noHtml -outfile ${rptDir}/invs_summary_updated.rpt
report_power -outfile ${rptDir}/invs_power_updated.rpt
report_timing -path_type full_clock > ${rptDir}/invs_timing_updated.rpt
report_power -leakage -outfile ${rptDir}/invs_leakage_power_updated.rpt
timeDesign -postRoute >  ${rptDir}/invs_timeDesign_updated.rpt
report_power -view POWER_VIEW -outfile ${rptDir}/invs_ff_power_updated.rpt
report_power -view WC_VIEW -outfile ${rptDir}/invs_ss_power_updated.rpt
write_sdc > ${rptDir}/invs_route_updated.sdc

exit