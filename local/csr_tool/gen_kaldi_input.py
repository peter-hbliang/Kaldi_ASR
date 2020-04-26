import os
import sys

assert(len(sys.argv)==2,'Usage: gen_kaldi_input.py input_list')

file_path=sys.argv[1]

word_list=[]
sentence_list=[]
story_list=[]

word_text=[]
sentence_text=[]
story_text=[]

with open('words.txt') as fp:
    for line in fp:
        word_list.append(line.strip())
with open('sentences.txt') as fp:
    for line in fp:
        sentence_list.append(line.strip())
with open('story.txt') as fp:
    for line in fp:
        story_list.append(line.strip())
with open('word_text_pying.txt') as fp:
    for line in fp:
        word_text.append(line.strip())
with open('sentences_text_pying.txt') as fp:
    for line in fp:
        sentence_text.append(line.strip())
with open('story_text_pying.txt') as fp:
    for line in fp:
        story_text.append(line.strip())

with open(file_path) as fp:
    for line in fp:
        file_name=line.split('/')
        type_name=file_name[-1].split('_')
        name=type_name[1]+'.wav'
        temp=file_name[-1].split('.')
        newline=temp[0]
        if name in word_list:
            idx=word_list.index(name)
            print(newline,word_text[idx])
        if name in sentence_list:
            idx=sentence_list.index(name)
            print(newline,sentence_text[idx])
        if name in story_list:
            idx=story_list.index(name)
            print(newline,story_text[idx])
