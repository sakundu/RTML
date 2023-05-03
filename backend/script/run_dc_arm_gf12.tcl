sh date
set_host_options -max_cores 4

set top_module accelerator
set libdir "/home/zf4_projects/RTML/sakundu/PNR/REF/gf12/db"

lappend search_path "."
lappend search_path "/home/tool/synopsys/DesignCompiler/N-2017.09-SP5/libraries/syn"
lappend search_path $libdir


set link_library {}
lappend link_library "sc9mcpp84_12lp_base_rvt_c14_sspg_sigcmax_max_0p72v_125c.db"
lappend link_library "sc9mcpp84_12lp_base_lvt_c14_sspg_sigcmax_max_0p72v_125c.db"
lappend link_library "sc9mcpp84_12lp_base_slvt_c14_sspg_sigcmax_max_0p72v_125c.db"

set target_library {}
lappend target_library "sc9mcpp84_12lp_base_rvt_c14_sspg_sigcmax_max_0p72v_125c.db"
lappend target_library "sc9mcpp84_12lp_base_lvt_c14_sspg_sigcmax_max_0p72v_125c.db"
lappend target_library "sc9mcpp84_12lp_base_slvt_c14_sspg_sigcmax_max_0p72v_125c.db"

set synthetic_library {}
lappend synthetic_library "dw_foundation.sldb"

set symbol_library {}
lappend symbol_library "generic.sdb"

set link_path [concat $link_library $synthetic_library]

set wire_load_model ""
set wire_load_mode enclosed
set timing_use_enhanced_capacitance_modeling true

set dont_use_cells 1
set dont_use_cell_list ""


# Start
remove_design -all

if {[file exists template]} {
	exec rm -rf template
}
exec mkdir template


if {![file exists gate]} {
	exec mkdir gate
}

if {![file exists rpt]} {
	exec mkdir rpt
}

# Compiler drectives
set compile_effort   "high"

#set compile_flatten_all 1
set compile_no_new_cells_at_top_level false
set hdlin_enable_vpp true
set hdlin_auto_save_templates false

if {[regexp "viterbi" $top_module]} {
	set design_lib dec_viterbi
} else {
	set design_lib WORK
}

define_design_lib $design_lib -path template
set verilogout_single_bit false

# read RTL
source for_dc.tcl

foreach rtl_file $rtl_all {
    if {[regexp "viterbi" $top_module]} {
		analyze -format vhdl -lib $design_lib $rtl_file
	} else {
		#analyze -format sverilog -lib $design_lib $rtl_file
		analyze -format verilog -lib $design_lib $rtl_file
	}
}

elaborate $top_module -lib $design_lib -update
current_design $top_module

# Link Design
set dc_shell_status [ link ]
if {$dc_shell_status == 0} {
	echo "****************************************************"
	echo "* ERROR!!!! Failed to Link...exiting prematurely.  *"
	echo "****************************************************"
	exit
}

# read_sdc will only read fully expanded sdc file
# Default SDC Constraints
source -v -e  ${top_module}.sdc

current_design $top_module

set_cost_priority {max_transition max_fanout max_delay max_capacitance}
set_fix_multiple_port_nets -all -buffer_constants
set_fix_hold [all_clocks]


# More compiler directives
set compile_effort   "high"
set_app_var ungroup_keep_original_design true
set_register_merging [get_designs $top_module] false
set compile_seqmap_propagate_constants false
set compile_seqmap_propagate_high_effort false
foreach_in_collection design [ get_designs "*" ] {
	current_design $design
	set_fix_multiple_port_nets -all
}

current_design $top_module

set dc_shell_status [ compile_ultra -scan -no_autoungroup -timing_high_effort_script  -exact_map ]

if {$dc_shell_status == 0} {
	echo "*******************************************************"
	echo "* ERROR!!!! Failed to compile...exiting prematurely.  *"
	echo "*******************************************************"
	exit
}

sh date

current_design $top_module
define_name_rules verilog -remove_internal_net_bus -remove_port_bus
change_names -rules verilog -hierarchy

if {[info exists use_physopt] && ($use_physopt == 1)} {
	write -format verilog -hier -output [format "%s%s%s" gate/ $top_module _hier_fromdc.v]
} else {
	write -format verilog -hier -output [format "%s%s%s" gate/ $top_module .v]
}

current_design $top_module
write_sdc [format "%s%s%s" gate/ $top_module .sdc]

# Write Reports
redirect [format "%s%s%s" rpt/ $top_module _qor.rep] { report_qor }
redirect [format "%s%s%s" rpt/ $top_module _area.rep] { report_area }
redirect -append [format "%s%s%s" rpt/ $top_module _area.rep] { report_reference }
redirect [format "%s%s%s" rpt/ $top_module _cell.rep] { report_cell }
redirect [format "%s%s%s" rpt/ $top_module _design.rep] { report_design }
redirect [format "%s%s%s" rpt/ $top_module _power.rep] { report_power }
redirect [format "%s%s%s" rpt/ $top_module _timing.rep] \
  { report_timing -path full -max_paths 100 -nets -transition_time -capacitance -significant_digits 3}
redirect [format "%s%s%s" rpt/ $top_module _check_timing.rep] { check_timing }
redirect [format "%s%s%s" rpt/ $top_module _check_design.rep] { check_design }


set inFile  [open rpt/$top_module\_area.rep]
while { [gets $inFile line]>=0 } {
    if { [regexp {Total cell area:} $line] } {
        set AREA [lindex $line 3]
    }
}
close $inFile
set inFile  [open rpt/$top_module\_power.rep]
while { [gets $inFile line]>=0 } {
    if { [regexp {Total Dynamic Power} $line] } {
        set PWR [lindex $line 4]
    } elseif { [regexp {Cell Leakage Power} $line] } {  
        set LEAK [lindex $line 4] 
    }
}
close $inFile

set path    [get_timing_path -nworst 1]
set WNS     [get_attribute $path slack]

set outFile [open result_dc.rpt w]
puts $outFile "$AREA\t$WNS\t$PWR\t$LEAK"
close $outFile

## Create cellmaster file
set cellList [get_cells -hier * -filter "is_hierarchical==false"]
set outfp [open ${top_module}_cell.master "w"]
foreach_in_collection cell $cellList {
    set master [get_attribute -quiet $cell ref_name]
    set name   [get_attribute -quiet $cell full_name]
    if { [regexp {Logic} $name] } { continue }
    if { [regexp {Mem} $name] } {
        puts $outfp "$name $master"
    }
}
close $outfp

# Check Design and Detect Unmapped Design
set unmapped_designs [get_designs -filter "is_unmapped == true" $top_module]
if {  [sizeof_collection $unmapped_designs] != 0 } {
	echo "****************************************************"
	echo "* ERROR!!!! Compile finished with unmapped logic.  *"
	echo "****************************************************"
	exit
}
echo "run.scr completed successfully"

write_sdc [format "%s%s%s" rpt/ ${top_module}_updated .sdc]

set clock_period [get_attribute [get_clocks] period]

set clock_period [expr $clock_period - $WNS]

#exec sed -i "s/__clock__/$clock_period/g" ./${top_module}_template.sdc

exec sed -i "s/period/period@/g" rpt/${top_module}_updated.sdc
exec sed -i "s/@.*//g" rpt/${top_module}_updated.sdc
exec sed -i "s/period/period  ${clock_period}/g" rpt/${top_module}_updated.sdc

read_sdc rpt/${top_module}_updated.sdc

# Write Reports
redirect [format "%s%s%s" rpt/ $top_module _qor_updated.rep] { report_qor }
redirect [format "%s%s%s" rpt/ $top_module _area_updated.rep] { report_area }
redirect -append [format "%s%s%s" rpt/ $top_module _area_updated.rep] { report_reference }
redirect [format "%s%s%s" rpt/ $top_module _cell_updated.rep] { report_cell }
redirect [format "%s%s%s" rpt/ $top_module _design_updated.rep] { report_design }
redirect [format "%s%s%s" rpt/ $top_module _power_updated.rep] { report_power }
redirect [format "%s%s%s" rpt/ $top_module _timing_updated.rep] \
  { report_timing -path full -max_paths 100 -nets -transition_time -capacitance -significant_digits 3}
redirect [format "%s%s%s" rpt/ $top_module _check_timing_updated.rep] { check_timing }
redirect [format "%s%s%s" rpt/ $top_module _check_design_updated.rep] { check_design }

write_parasitics -output gate/${top_module}_updated.spef

set inFile  [open rpt/$top_module\_area_updated.rep]
while { [gets $inFile line]>=0 } {
    if { [regexp {Total cell area:} $line] } {
        set AREA [lindex $line 3]
    }
}
close $inFile
set inFile  [open rpt/$top_module\_power_updated.rep]
while { [gets $inFile line]>=0 } {
    if { [regexp {Total Dynamic Power} $line] } {
        set PWR [lindex $line 4]
    } elseif { [regexp {Cell Leakage Power} $line] } {
        set LEAK [lindex $line 4] 
    }
}
close $inFile

set path    [get_timing_path -nworst 1]
set WNS     [get_attribute $path slack]

set outFile [open result_dc_updated.rpt w]
puts $outFile "$AREA\t$WNS\t$PWR\t$LEAK"
close $outFile

exit
