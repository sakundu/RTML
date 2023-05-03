set clock_cycle xx_cp_xx 
set uncertainty 0
set io_delay 0

set clock_port clk 

create_clock -name clk -period $clock_cycle [get_ports $clock_port]
set_clock_uncertainty $uncertainty [all_clocks]

set_input_delay -clock [get_clocks clk] -add_delay -max $io_delay   [get_ports [remove_from_collection [all_inputs]  [get_attribute [all_clocks] sources]]]
set_output_delay -clock [get_clocks clk] -add_delay -max $io_delay   [get_ports [remove_from_collection [all_outputs] [get_attribute [all_clocks] sources]]]

