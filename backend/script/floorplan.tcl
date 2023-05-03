#########################
# floorplan
setOptMode -powerEffort high -leakageToDynamicRatio 0.5

setGenerateViaMode -auto true
generateVias

createBasicPathGroups -expanded

if {[info exist ::env(UTIL)] && $::env(UTIL) > 0 } {
    set target_utilization $::env(UTIL)
    floorPlan -site $site -r 1.0 $target_utilization 5.0 5.0 5.0 5.0
} else {
    floorPlan -site $site -r 1.0 0.7 5.0 5.0 5.0 5.0
}

defOut -routing ${encDir}/${design}_floorplan_auto.def
saveNetlist ${encDir}/${design}_floorplan_auto.v
saveDesign ${encDir}/${design}_floorplan_auto.enc
