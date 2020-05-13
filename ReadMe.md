# Kaldi_ASR
* 使用kaldi訓練大詞彙繁體中文語音辨識器。
    * [Formosa Speech Recognition Challenge Workshop 2018 paper](https://drive.google.com/file/d/15inv3RHf9bTxwhwqrXwWbqNcxAfDgoxl/view)
## Dependency
* kaldi-asr:
    * https://github.com/kaldi-asr/kaldi

## Usage
  ### Download pre-trained LM
 
```
bash data.sh
```
  ### Preprocess
```
bash utils/parse_options.sh
bash preprocess.sh 3 wav/tcc300 data/tcc300
```
1. 將你要辨識的音檔資料夾，放進wav這個資料夾裡。
1. 第一個位置請放num_of_jobs(num_of_jobs請勿超過語者數量)。
2. 第二個位置請放特徵存放位置的資料夾。
3. 抽好的特徵參數將會被放到data裡。
## Decode
```
bash decode.sh 1 data/tcc300 decode-tcc300 decode-tcc300_lm
```
  
1. 第一個位置一樣請決定number of jobs(decode 是使用cpu，每個job會占用約7~9G的RAM)。
2. 第二個位置請放剛剛抽好的特徵參數的資料夾位置。
3. 第三個位置請幫第一階段的output取一個資料夾名字。
4. 第四個位置請幫第二階段的output取一個資料夾名字。

## Result
讀取結果
1. 辨識結果將存放在result資料夾底下。
2. 檔案前面的數字代表LM的權重，數字越大，代表辨識器越相信LM。
