# extras/.env.sh
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth1   # or your inter-node iface
export MASTER_PORT=29500
# MASTER_ADDR can be the first entry in hostfile:
export MASTER_ADDR=10.65.4.2

