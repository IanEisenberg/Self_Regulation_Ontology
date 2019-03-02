#!/bin/bash
#
# SLURM control file automatically created by launch
#SBATCH -N 10
#SBATCH -J irtfa       # Job Name
#SBATCH -o irtfa.o%j # Name of the output file (eg. myMPI.oJobID)
#SBATCH -p normal
#SBATCH -t 24:00:00
#SBATCH -n 10 
#SBATCH -A Analysis_Lonestar
#----------------
# Job Submission
#----------------
module load Rstats
export TACC_LAUNCHER_PPN=1
export EXECUTABLE=$TACC_LAUNCHER_DIR/init_launcher
export WORKDIR=.
export CONTROL_FILE=run_all_irtfa.sh
export TACC_LAUNCHER_SCHED=interleaved
cd $WORKDIR
echo " WORKING DIR:   $WORKDIR/"
$TACC_LAUNCHER_DIR/paramrun SLURM $EXECUTABLE $WORKDIR $CONTROL_FILE
echo " "
echo " Parameteric Job Complete"
echo " "
