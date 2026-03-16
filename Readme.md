## Run：

cd /mnt/server/WMFactory/frontend
/mnt/server/WMFactory/venvs/diamond/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8081


## Todo:

1. 自动退出功能

2. 数据集的适配

3. 每一个再确认一遍，分析慢的原因和原始参数的区别。

4. 加速！！！注意有Text Encoder的模型怎么优化，注意输入的分辨率，注意ddim steps的设置。

5. 是不是load到了cpu上之类的问题