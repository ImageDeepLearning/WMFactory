# Diamond Model Service

独立 DIAMOND 模型服务（在 DIAMOND 对应环境中运行），供网关 `frontend/server.py` 通过 HTTP 调用。

## 启动

```bash
cd /mnt/server/WMFactory/services/diamond
/mnt/server/WMFactory/venvs/diamond/bin/python -m uvicorn app:app --host 127.0.0.1 --port 9001
```

## 接口

- `POST /health`
- `POST /load`
- `POST /sessions/start`
- `POST /sessions/step`
- `POST /sessions/reset`
- `POST /datasets/random-image`

## 说明

- 该服务在本进程内持有模型与会话状态。
- 当前单会话语义：每次 `/sessions/start` 会刷新 `session_id`。
