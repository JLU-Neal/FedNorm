#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=<zt520> # required to send email notifcations - please replace <your_username> with your college login name or email address

MODEL=graphsage
FL_ALG=FedAvg
DATASET=sider
DISTRIBUTION=homo
WORKER_NUM=1

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

source /vol/bitbucket/{USER}/miniconda/etc/profile.d/conda.sh
conda activate fedgraphnn
. /vol/cuda/11.1.0-cudnn8.0.4.30/setup.sh

export I_MPI_PROCESS_MANAGER=mpd
mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 main_fedavg.py \
  --model $MODEL \
  --fl_algorithm $FL_ALG \
  --dataset $DATASET \
  --partition_method $DISTRIBUTION  \

uptime