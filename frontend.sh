
# conda init
# conda activate /home/mengfei/miniconda3/envs/WorldFM
export LD_LIBRARY_PATH="/home/mengfei/miniconda3/envs/WorldFM/lib:$LD_LIBRARY_PATH"
cd frontend
/home/mengfei/miniconda3/envs/WorldFM/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8080