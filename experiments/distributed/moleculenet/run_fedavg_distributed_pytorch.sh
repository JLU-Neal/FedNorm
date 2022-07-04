#!/usr/bin/env bash

# CLIENT_NUM=$1 # 6
# WORKER_NUM=$2 # 1
# SERVER_NUM=$3 # 1
# GPU_NUM_PER_SERVER=$4 # 1
# MODEL=$5 # graphsage
# DISTRIBUTION=$6 # homo
# PARTITION_ALPHA=$7 # 0.2
# ROUND=$8 # 150
# EPOCH=$9 # 1
# BATCH_SIZE=$10 # 1
# LR=${11} # 0.0015
# HIDDEN_DIM=${12} # 256
# NODE_DIM=${13} # 256
# DR=${14} # 0.3
# READ_DIM=${15} # 256
# GRAPH_DIM=${16} # 256
# DATASET=${17} # sider
# FL_ALG=${18} # FedAvg

MODEL=$1 # graphsage
FL_ALG=${2} # FedAvg
DATASET=${3} # sider
DISTRIBUTION=$4 # homo
IS_DATA_SHARING=$5 
IS_FEDNORM=$6
WORKER_NUM=$7

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 main_fedavg.py \
  --model $MODEL \
  --fl_algorithm $FL_ALG \
  --dataset $DATASET \
  --partition_method $DISTRIBUTION  \
  --client_num_per_round $WORKER_NUM \
  --is_data_sharing $IS_DATA_SHARING \
  --SetNet $IS_FEDNORM \
# mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 main_fedavg.py \
#   --gpu_server_num $SERVER_NUM \
#   --gpu_num_per_server $GPU_NUM_PER_SERVER \
#   --model $MODEL \
#   --dataset $DATASET \
#   --hidden_size $HIDDEN_DIM \
#   --node_embedding_dim $NODE_DIM \
#   --dropout $DR \
#   --readout_hidden_dim $READ_DIM \
#   --graph_embedding_dim $GRAPH_DIM \
#   --partition_method $DISTRIBUTION  \
#   --partition_alpha $PARTITION_ALPHA \
#   --client_num_in_total $CLIENT_NUM \
#   --client_num_per_round $WORKER_NUM \
#   --comm_round $ROUND \
#   --epochs $EPOCH \
#   --fl_algorithm $FL_ALG \
#   --batch_size $BATCH_SIZE \
#   --lr $LR 
