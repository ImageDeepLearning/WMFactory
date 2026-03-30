# WMFactory Frontend Gateway

该目录是统一网关（Gateway），不直接运行模型推理。

- 前端页面：`frontend/web/`
- 网关 API：`frontend/server.py`
- 网关适配器：`frontend/adapters/`

当前 DIAMOND 模式：

- 模型服务代码：`services/diamond/app.py`
- 网关默认懒启动：点击“加载模型”时，若服务未运行，网关会自动启动服务再执行加载

当前 WorldFM 模式：

- 模型服务代码：`services/worldfm/app.py`
- 网关同样按“确认加载”懒启动服务

## 最简启动

只需启动网关：

```bash
conda activate /home/mengfei/miniconda3/envs/WorldFM
cd /mnt/server/WMFactory/frontend
/mnt/server/WMFactory/venvs/diamond/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8080

conda activate /home/mengfei/miniconda3/envs/WorldFM
export LD_LIBRARY_PATH="/home/mengfei/miniconda3/envs/WorldFM/lib:$LD_LIBRARY_PATH"
cd frontend
/home/mengfei/miniconda3/envs/WorldFM/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8080

conda activate /home/mengfei/miniconda3/envs/WorldFM
export LD_LIBRARY_PATH="/home/mengfei/miniconda3/envs/WorldFM/lib:$LD_LIBRARY_PATH"
/home/mengfei/miniconda3/envs/WorldFM/bin/python -m uvicorn worldfmEnd.server:app --host 0.0.0.0 --port 8889
```

然后打开：

- `http://127.0.0.1:8080`

## 可选：手动启动模型服务

```bash
cd /mnt/server/WMFactory/services/diamond
/mnt/server/WMFactory/venvs/diamond/bin/python -m uvicorn app:app --host 127.0.0.1 --port 9001

cd /home/mengfei/WMFactory/services/worldfm
/home/mengfei/miniconda3/envs/WorldFM/bin/python -m uvicorn app:app --host 127.0.0.1 --port 9002
```

## 关键环境变量

- `WM_DIAMOND_URL`：服务地址（默认 `http://127.0.0.1:9001`）
- `WM_DIAMOND_AUTOSTART`：是否自动启动服务（默认 `1`）
- `WM_DIAMOND_PYTHON`：启动服务使用的 Python（默认 `venvs/diamond/bin/python`）
- `WM_DIAMOND_SERVICE_DIR`：模型服务工作目录（默认 `services/diamond`）
- `WM_DIAMOND_LOG`：模型服务日志路径（默认 `services/diamond/diamond_service.log`）
- `WM_DIAMOND_STARTUP_TIMEOUT`：服务启动等待秒数（默认 `240`）
- `WM_MODEL_HTTP_TIMEOUT`：网关请求模型服务超时秒数（默认 `120`）
- `WM_DIAMOND_STEP_LOG_EVERY`：`/sessions/step` 日志节流间隔（默认 `20`）

DIAMOND 服务侧可用：

- `DIAMOND_CKPT_PATH`
- `DIAMOND_SPAWN_DIR`

WorldFM 相关：

- `WM_WORLDFM_URL`：服务地址（默认 `http://127.0.0.1:9002`）
- `WM_WORLDFM_AUTOSTART`：是否自动启动服务（默认 `1`）
- `WM_WORLDFM_PYTHON`：启动服务使用的 Python（默认 `venvs/worldfm/bin/python`）
- `WM_WORLDFM_SERVICE_DIR`：模型服务工作目录（默认 `services/worldfm`）
- `WM_WORLDFM_LOG`：模型服务日志路径（默认 `services/worldfm/worldfm_service.log`）
- `WM_WORLDFM_STARTUP_TIMEOUT`：服务启动等待秒数（默认 `240`）
- `WM_WORLDFM_STEP_LOG_EVERY`：`/sessions/step` 日志节流间隔（默认 `20`）
- `WM_WORLDFM_CONFIG`：可选，自定义 worldfm 配置 YAML
- `WM_WORLDFM_GPU_INDEX`：推理 GPU 序号
- `WM_HF_ENDPOINT`：HF 镜像端点（默认 `https://hf-mirror.com`）

WonderWorld 相关：

- `WM_WONDERWORLD_URL`：服务地址（默认 `http://127.0.0.1:9004`）
- `WM_WONDERWORLD_AUTOSTART`：是否自动启动服务（默认 `1`）
- `WM_WONDERWORLD_SERVICE_PYTHON`：启动服务使用的 Python（默认 `venvs/WonderWorld/bin/python`）
- `WM_WONDERWORLD_SERVICE_DIR`：模型服务工作目录（默认 `services/wonderworld`）
- `WM_WONDERWORLD_LOG`：模型服务日志路径（默认 `services/wonderworld/wonderworld_service.log`）
- `WM_WONDERWORLD_STARTUP_TIMEOUT`：服务启动等待秒数（默认 `240`）
- `WM_WONDERWORLD_STEP_LOG_EVERY`：`/sessions/step` 日志节流间隔（默认 `20`）

WHAM 相关：

- `WM_WHAM_URL`：服务地址（默认 `http://127.0.0.1:9007`）
- `WM_WHAM_AUTOSTART`：是否自动启动服务（默认 `1`）
- `WM_WHAM_PYTHON`：启动服务使用的 Python（默认 `venvs/wham/bin/python`）
- `WM_WHAM_SERVICE_DIR`：模型服务工作目录（默认 `services/wham`）
- `WM_WHAM_LOG`：模型服务日志路径（默认 `services/wham/wham_service.log`）
- `WM_WHAM_STARTUP_TIMEOUT`：服务启动等待秒数（默认 `180`）
- `WM_WHAM_STEP_LOG_EVERY`：`/sessions/step` 日志节流间隔（默认 `20`）
- `WM_WHAM_MODEL_PATH`：WHAM checkpoint 路径（默认 `models/wham/models/WHAM_200M.ckpt`）
- `WM_WHAM_GPU_INDEX`：推理 GPU 序号（默认 `0`）

## 网关 API

- `GET /api/models`
- `POST /api/models/load`
- `GET /api/datasets`
- `POST /api/datasets/random-image`
- `POST /api/sessions/start`
- `POST /api/sessions/step`
- `POST /api/sessions/reset`
- `POST /api/sessions/progress`

详细接口规范见根目录 `MUSTSEE.md`。
