#!/usr/bin/env bash

NPBOS_DIR=$1
MASS_NUM=$2
N_NU=$3
N_PI=$4
EPA=$5
KAPPA=$6
CHI_PI=$7
CHI_NU=$8
ELEMENT=$9

cd "$NPBOS_DIR"

./npbos <<EOF
 &N  NCUT=100,   IEX =1,
     LAUTO = 0, 2, 4, 6,
     NEIGA = 2, 1, 1, 1,
     NDUPTA= 15, 15, 15, 15, 15, 15, 15,
     IWCF= 2,  NPSTW= 0,
 &END
$MASS_NUM $ELEMENT
   $N_NU   $N_PI
    0
 &INPT
      ED   = $EPA,
      RKAP = $KAPPA,
      CHN  = $CHI_NU,
      CHP  = $CHI_PI,

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
            first_2=$(echo "$LINE" | cut -c22-28)
        elif [ "$LINE_SUB" = "4  +  ( 1)" ] ; then
            first_4=$(echo "$LINE" | cut -c22-28)
        elif [ "$LINE_SUB" = "6  +  ( 1)" ] ; then
            first_6=$(echo "$LINE" | cut -c22-28)
        elif [ "$LINE_SUB" = "0  +  ( 2)" ] ; then
            second_0=$(echo "$LINE" | cut -c22-28)
        fi
    fi
done < "$FILE"

echo "$first_2 $first_4 $first_6 $second_0"