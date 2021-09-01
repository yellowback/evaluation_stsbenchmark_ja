
## これはなに

自動翻訳版stsbenchmarkの日本語評価用のスクリプトです。

## パッケージインストール

```
$ pip install torch==1.8.1
$ pip install -r requirements.txt
```

## データダウンロード

```
$ cd stsb_multi_mt_ja
$ wget https://github.com/PhilipMay/stsb-multi-mt/raw/main/data/stsb-ja-test.csv
$ wget https://github.com/PhilipMay/stsb-multi-mt/raw/main/data/stsb-en-test.csv
$ cd ..
```

## 使い方

### 日本語 - 日本語

```
# sentence transformers
$ python evaluation_stsbenchmark_ja.py

# mUSE
$ python evaluation_stsbenchmark_muse_ja.py
```

### 日本語 - 英語

```
# sentence transformers
$ python evaluation_stsbenchmark_ja_en.py

# mUSE
$ python evaluation_stsbenchmark_muse_ja_en.py
```


