# Memory

## DIAMOND

### Repository
- Path: `/mnt/server/WMFactory/models/diamond`
- Branch for CSGO demo: `csgo`

### Environment
- Conda env path: `/mnt/server/WMFactory/venvs/diamond`
- Python version: `3.10`
- Recommended conda package cache:
  - `CONDA_PKGS_DIRS=/mnt/server/WMFactory/venvs/.conda_pkgs`

### Install (with stable network settings)
```bash
cd /mnt/server/WMFactory/models/diamond
http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= \
/mnt/server/WMFactory/venvs/diamond/bin/pip install \
-i https://pypi.tuna.tsinghua.edu.cn/simple \
--trusted-host pypi.tuna.tsinghua.edu.cn \
-r requirements.txt -v
```

### Run CSGO Demo
```bash
cd /mnt/server/WMFactory/models/diamond
git checkout csgo
http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= HF_ENDPOINT=https://hf-mirror.com \
/mnt/server/WMFactory/venvs/diamond/bin/python src/play.py
```

### First-Run Notes
- First launch will download model assets from Hugging Face (including a large `csgo.pt`, around 1.5G).
- Typical success logs:
  - `Using cuda:0 for rendering.`
  - keymap/controls printed
  - `Press enter to start`

### Runtime Notes
- In non-interactive terminals, `Press enter to start` may raise `EOFError`.
- Use an interactive terminal session to actually play (`WASD`, arrows, etc.).
