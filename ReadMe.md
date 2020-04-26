1.download LM
  a.使用./data.sh將LM載下來

2.preprocessing(usage:./preprocessiong.sh wav/tcc300 data/tcc300)
  a.首先將你要辨識的音檔資料夾，放進wav這個資料夾裡。ex:wav/tcc300/{1.wav,2.wav,......}
  b.執行preprocessing.sh檔。第一個位置請放音檔位置的資料夾，第二個位置請放特徵存放位置的資料夾
    ex:./preprocessiong.sh wav/tcc300 data/tcc300
  c.抽好的特徵參數將會被放到data裡。ex:data/tcc300/{fbank,text,utt2spk,......}  

3.decode(usage:./decode.sh 3 data/tcc300 decode-tcc300 decode-tcc300_lm)
  a.執行decode.sh檔，
    第一個位置請決定number of jobs(decode 是使用cpu，每個job會占用約7~9G的RAM)
    第二個位置請放剛剛抽好的特徵參數的資料夾位置
    第三個位置請幫第一階段的output取一個資料夾名字
    第四個位置請幫第二階段的output取一個資料夾名字
    ex:./decode.sh 3 data/tcc300 decode-tcc300 decode-tcc300_lm

4.讀取結果
  a.辨識結果將存放在result資料夾底下。ex:result/decode-tcc300_lm/{1.txt,2.txt,......}
  b.檔案前面的數字代表LM的權重，數字越大，代表辨識器越相信LM。
