#! /bin/bash

#$ -t 1-20
#$ -N decision-making

#chmod u+x submit.sh

python afs/ir/users/g/i/gilsho/Desktop/Classes/BIOE332/bioe332assign2/assign2.py -k $SGE_TASK_ID
