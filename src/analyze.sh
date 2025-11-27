#!/usr/bin/env bash

NPBOS_DIR=$1
MASS_NUM=$2
N_NU=$3
eps=$4
kappa=$5
chi_pi=$6
chi_n=$7

cd "$NPBOS_DIR"

./npbos <<EOF
 &N  NCUT=100,   IEX =1,
     LAUTO = 0, 2, 4, 6,
     NEIGA = 2, 1, 1, 1,
     NDUPTA= 8, 8, 8, 8, 8, 8, 8,
     IWCF= 2,  NPSTW= 0,
 &END
$MASS_NUM Sm
   $N_NU   6
    0
 &INPT
      ED   = $eps,
      RKAP = $kappa,
      CHN  = $chi_n,
      CHP  = $chi_pi,

 &END
 
E
EOF

sleep 0.3

FILE="./out1.dat"

COL=$(wc -l < "$FILE")

while read LINE; do
    COL=$((COL - 1))
    if [ ${COL} -lt 9 ] ; then
        LINE_SUB=$(echo "$LINE" | cut -c7-16)
        if [ "$LINE_SUB" = "2  +  ( 1)" ] ; then
            first_2=$(echo "$LINE" | cut -c23-28)
        elif [ "$LINE_SUB" = "4  +  ( 1)" ] ; then
            first_4=$(echo "$LINE" | cut -c23-28)
        elif [ "$LINE_SUB" = "6  +  ( 1)" ] ; then
            first_6=$(echo "$LINE" | cut -c23-28)
        elif [ "$LINE_SUB" = "0  +  ( 2)" ] ; then
            second_0=$(echo "$LINE" | cut -c23-28)
        fi
    fi
done < "$FILE"

echo "$first_2 $first_4 $first_6 $second_0"