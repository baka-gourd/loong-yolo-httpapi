# 龙 API

### 安装

1. 下载代码

   ```bash
   git clone https://github.com/baka-gourd/loong-yolo-httpapi
   ```
2. 安装依赖

   - 安装 [PyTorch](https://pytorch.org/get-started/)
   - 执行 `pip install -r requirements.txt`
3. 下载模型

   ```bash
   ...
   ```

### 使用

1. 启动服务

   ```bash
   python main.py
   ```
2. 发送 post 包至 `http://localhost:8008/pics`

   - 参数

     - `pic_base64`: base64 编码的图片
     - `pic_url`: 图片的 url 地址
     - `accuracy`: 检测精度，取值范围为 0.0 ~ 1.0，默认为 0.8
   - 返回

     - `loong`: 检测结果

       ```json
       {
         "result": true
       }
       ```

         - `result`: 检测结果，true 为检测到龙，false 为未检测到龙

### 测试

```bash
curl -X 'POST' \
  'http://127.0.0.1:8008/pics/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
   "pic_url": "https://img.sszg.com/article/contents/2022/08/12/small_20220812105342529.jpeg"
}'
```
