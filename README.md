# MC_research_2 (project2)

## 概要
このプロジェクトは、ニューラルネットワークを用いて原子核物理学における **IBM2 (Interacting boson model 2)** のハミルトニアンのパラメータ ($\epsilon, \kappa, \chi_\nu, \chi_pi$) と PES(Potential energy surface)のスケール因子 ($C_\beta$)を推定するための研究用コードです。
**Hartree-Fock-Bogoliubov(HFB)** 法などで計算されたPESから、最適なIBM2パラメータを逆推定することを目的としています。また、推定されたパラメータを用いてIBMの計算コードである`NPBOS`を実行し、エネルギースペクトルを計算することで実験値と比較評価する機能も備えています。

## ディレクトリ構造

```
project2/
├── configs/            # 各種設定
├── data/               # データセット
│     ├ raw/
│     └ processed/
├── outputs/            # 学習結果, 評価結果など
├── src/                # ソースコード
│     ├ dataset.py
│     ├ evaluate.py
│     ├ model.py
│     ├ utils.py
│     └ visualize.py
├── scripts/
│     ├ analyze.py
│     ├ train.py
│     └ plot.py
├── requirements.txt    # Python依存ライブラリ
```