# WMArena

WMArena 是独立于原有 `frontend/` 的双模型竞技场入口，所有新增代码都位于本目录。

## 启动

```bash
cd /mnt/server/WMFactory/WMArena
http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= /mnt/server/WMFactory/venvs/diamond/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8081
```

打开 `http://127.0.0.1:8081`。

## 说明

- 复用原有 `frontend/adapters/` 与模型服务协议
- 不修改原有 `frontend/server.py` 和原主前端
- 左右两侧会使用独立端口与独立 GPU 进行模型加载
- 第一阶段强制左右选择不同模型
