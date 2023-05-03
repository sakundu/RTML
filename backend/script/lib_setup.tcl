# This script was written and developed by ABKGroup students at UCSD. However, the underlying commands and reports are copyrighted by Cadence. 
# We thank Cadence for granting permission to share our research to help promote and foster the next generation of innovators.
# lib and lef, RC setup

set tmp_dir "/home/zf4_projects/RTML/sakundu/PNR/REF/gf12/"
set libdir "${tmp_dir}/lib"
set lefdir "${tmp_dir}/lef"
set qrcdir "${tmp_dir}/qrc"

set_db init_lib_search_path { \
    ${libdir} \
    ${lefdir} \
}

set libworst "  
    ${libdir}/sc9mcpp84_12lp_base_lvt_c14_sspg_sigcmax_max_0p72v_125c.lib \
    ${libdir}/sc9mcpp84_12lp_base_rvt_c14_sspg_sigcmax_max_0p72v_125c.lib \
    ${libdir}/sc9mcpp84_12lp_base_slvt_c14_sspg_sigcmax_max_0p72v_125c.lib \
    "

set libpower "
    ${libdir}/sc9mcpp84_12lp_base_lvt_c14_ffpg_sigcmin_min_0p88v_125c.lib \
    ${libdir}/sc9mcpp84_12lp_base_rvt_c14_ffpg_sigcmin_min_0p88v_125c.lib \
    ${libdir}/sc9mcpp84_12lp_base_slvt_c14_ffpg_sigcmin_min_0p88v_125c.lib \
    "

set libbest " 
    ${libdir}/sc9mcpp84_12lp_base_lvt_c14_ffpg_sigcmin_min_0p88v_m40c.lib \
    ${libdir}/sc9mcpp84_12lp_base_rvt_c14_ffpg_sigcmin_min_0p88v_m40c.lib \
    ${libdir}/sc9mcpp84_12lp_base_slvt_c14_ffpg_sigcmin_min_0p88v_m40c.lib \
    "

set lef "
    ${lefdir}/12LP_13M_3Mx_2Cx_4Kx_2Hx_2Gx_LB_84cpp_tech.lef \
    ${lefdir}/sc9mcpp84_12lp_base_lvt_c14.lef \
    ${lefdir}/sc9mcpp84_12lp_base_rvt_c14.lef \
    ${lefdir}/sc9mcpp84_12lp_base_slvt_c14.lef \
    "

set qrc_max "${qrcdir}/SigCmax/qrcTechFile"
set qrc_min "${qrcdir}/SigCmax/qrcTechFile"
#
# Ensures proper and consistent library handling between Genus and Innovus
#set_db library_setup_ispatial true
setDesignMode -process 12
