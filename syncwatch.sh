#!/bin/bash
LOCAL_DIR="/home/saiga/rpl/remote_server/Mask2Former"
REMOTE_HOST="sairamm_hpc@marvin.hpc.uni-bonn.de"
REMOTE_PATH="/lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former"
RSYNC_OPTS="-avz --progress"

echo "=== Mask2Former Sync Monitor ==="
echo "Local:  ${LOCAL_DIR}/"
echo "Remote: ${REMOTE_HOST}:${REMOTE_PATH}/"
echo "================================"

# Ensure remote directory exists
ssh -p 22 ${REMOTE_HOST} "mkdir -p ${REMOTE_PATH}"

# Do an initial sync
echo "Performing initial sync..."
cd "${LOCAL_DIR}" && \
rsync ${RSYNC_OPTS} \
  --exclude={'.git','__pycache__','.vscode','data/raw','checkpoints','*.pyc','datasets','nohup.out'} \
  -e 'ssh -p 22' \
  . \
  "${REMOTE_HOST}:${REMOTE_PATH}/"
echo "Initial sync complete."
echo ""
echo "Watching for changes..."

while true; do
  inotifywait -r -e modify,create,delete,move \
    --exclude '\.git|__pycache__|\.vscode|data/raw|checkpoints|\.pyc|datasets|nohup\.out' \
    "${LOCAL_DIR}"
  
  sleep 2
  
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Syncing changes..."
  
  cd "${LOCAL_DIR}" && \
  rsync ${RSYNC_OPTS} \
    --exclude={'.git','__pycache__','.vscode','data/raw','checkpoints','*.pyc','datasets','nohup.out'} \
    -e 'ssh -p 22' \
    . \
    "${REMOTE_HOST}:${REMOTE_PATH}/"
  
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sync complete."
done
