# SConv

## Overview
https://qiita.com/3405691582/items/b41f8b37b99148e67ba9

## Usage
コンテナ上で操作します。
```
docker compose up -d
docker exec -it sconv bash
```
### Predict
モデルckptファイルを`sconv/weight`に配置し、`sconv/configs/predict/default.yaml`の`weight`欄を適宜変更してください。
```
make predict
```
で文章生成ができます

## Models
* 日本語Wikipedia500Mパラメータモデル: https://huggingface.co/myxy/SConv-Wiki-500M