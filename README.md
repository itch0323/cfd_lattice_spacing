# cfd_lattice_spacing
異なる格子数での流体シミュレーションを実行できます。

実行環境と使用方法は下記の通りです。

<br>

## 実行環境(env)
- Python 3.7.9
- Numpy 1.19.2
- Matplotlib 3.3.2
- cython 0.29.21

<br>

## 使用方法(usage)
1. リポジトリをクローンします。
```
git clone https://github.com/itch0323/cfd_lattice_spacing
cd cfd_lattice_spacing
```

<br>

2. 異なる格子数で流体シミュレーションを行うことができます。
```
python cfd.py 32
python cfd.py 128
```