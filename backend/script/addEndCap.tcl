setEndCapMode -reset
setEndCapMode -rightEdge ENDCAPTIE12_A9PP84TR_C14 \
              -topEdge TBCAPNWIN2_A9PP84TR_C14 \
              -bottomEdge TBCAPNWOUT2_A9PP84TR_C14 \
              -rightTopEdge INCNRCAPNWINTIE12_A9PP84TR_C14 \
              -rightBottomEdge INCNRCAPNWOUTTIE12_A9PP84TR_C14 \
              -rightBottomCorner CNRCAPNWOUTTIE12_A9PP84TR_C14 \
              -rightTopCorner CNRCAPNWINTIE12_A9PP84TR_C14 \
              -prefix BNDRY
addEndCap


addWellTap -cell FILLTIE11_A9PP84TR_C14  -maxGap 20 -checkerboard
