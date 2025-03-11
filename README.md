Feature extraction with Autoencoder
---

Python + PyTorch で CNTフォレストの構造解析を行うリポジトリです．  
異常構造の検知はオートエンコーダで，反射率予測はResNetで行いました．

# 使い方

## リポジトリのダウンロード
リポジトリをダウンロードするには、git を使っている方は `git clone <このリポジトリのurl>`で、それ以外の方は GitHub のこのページ上部にある緑色の「Code」というボタンから「Download ZIP」を押して zip ファイルをダウンロードしたのち解凍してください。

## 仮想環境の構築
> [!WARNING]
> このプロジェクトでは，pythonのバージョン管理に[pyenv](https://github.com/pyenv/pyenv)，パッケージと仮想環境の管理に[poetry](https://python-poetry.org/)を使っています．もし使っているパソコンに入っていなければそれぞれの installation または introduction を読んでインストールしてください．
>
> もし他のパッケージ管理ツールを使っている方（venv，pipenv，uv，condaなど）はそれに合うように，pyproject.toml を編集してください．

その後，ターミナル上でクローン/ダウンロードしたディレクトリに移動して次のコマンドを実行してください．

すると .venv という名前の新しいディレクトリが作成されて，pyproject.toml に書かれているパッケージがインストールされ仮想環境が構築されます．

```bash
poetry sync
```

仮想環境の中でプログラムを動かしたい場合には次の方法があります。

```bash
# 仮想環境の外から仮想環境を使う場合
poetry run python <pythonスクリプトへのパス>

# 仮想環境に入って使う場合
## 仮想環境に入る．poetry のバージョンによってコマンドが違うことに注意．
poetry shell # If poetry version < 2.0.0
poetry env activate # If poetry version >= 2.0.0

## プログラムの実行
python <pythonスクリプトへのパス>
```

> [!NOTE]
> pythonで仮想環境を使う意味は，必要なパッケージを必要な場所においておくためです．どういう状況でこの機能が欲しくなるのかは[この](https://www.python.jp/install/windows/venv.html#:~:text=Python%20%E3%82%92%E4%BD%BF,%E7%B4%B9%E4%BB%8B%E3%81%97%E3%81%BE%E3%81%99%E3%80%82)ページを参考にしてください．

## モデルの訓練と検証の実行
src/running_script ディレクトリの中に，モデルの訓練用と訓練終了後のモデルの検証用（特徴ベクトルの可視化や分類）のスクリプトが入っています．

# 実行時のメモ
学習には時間がかかるので研究室から離れていても実行させたい場合は、Linuxを使ってる人は `nohup` や `screen` を使ってssh接続を切った後でも学習を続けられて便利。

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

# 主要な依存関係
```bash
# pythonのバージョン
python == 3.10.11

# 機械学習パッケージ
scikit-learn >= 1.2.2
torch >= 2.0.1+cu118
torchvision >= 0.15.2+cu118

# 機械学習補助パッケージ
torchinfo >= 1.8.0
torcheval >= 0.0.7
torch-tb-profiler >= 0.4.3
tensorboard >= 2.18.0

# 他補助パッケージ
hydra-core >= 1.3.2
jupyter >= 1.0.0
matplotlib >= 3.9.3
japanize-matplotlib = 1.1.3
numpy >= 1.25.0
opencv-python = 4.8.0.74
polars >= 0.18.3
tqdm >= 4.65.0
umap-learn = 0.5.4
```
