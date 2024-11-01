Feature extraction with Autoencoder
---
# 実行時のメモ
学習には時間がかかるので、もし夜中に学習をさせたいときはLinuxを使ってる人は `nohup` や `screen` を使ってssh接続を切った後でも学習を続けられるようにしよう。
## `nohup`の使い方
バックグラウンドで実行
```bash
user@pc:~/project$ nohup python sample.py &
# 以下出力
[1] 12345
nohup: ~~~~
```
もし別のファイルに出力したければ次のように & の前にリダイレクトをする
```bash
user@pc:~/project$ nohup python sample.py > sample_log.txt &
```
実行されているかの確認は次のように
```bash
user@pc:~/project$ jobs
# 以下出力
[1]+ Running nohup python sample.py
```

## `screen`の使い方
バックグラウンドで実行
```bash
user@pc:~/project$ screen python sample.py
```
もし復帰したい場合は
```bash
user@pc:~/project$ screen -ls
# 以下出力
There are several suitable screens on:
    12345.~~~
# 出力ここまで
user@pc:~/project$ screen -r 1234
# 実行画面になる
```
