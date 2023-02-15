#!/bin/sh

if [ $# -ne 1 ]; then
	echo 'usage: ./download_model.sh [124M|355M|774M|1558M]' >&2
	exit 1
fi

mkdir -p "models/$1/"
for file in 'checkpoint' 'encoder.json' 'hparams.json' 'model.ckpt.data-00000-of-00001' 'model.ckpt.index' 'model.ckpt.meta' 'vocab.bpe'; do
	wget "https://openaipublic.blob.core.windows.net/gpt-2/models/$1/$file" -O "models/$1/$file"
done
