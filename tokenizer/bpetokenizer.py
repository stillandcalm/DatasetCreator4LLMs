import sentencepiece as spm
spm.SentencePieceTrainer.Train(
    '--input=data/all.txt '
    '--model_prefix=llama8b_tokenizer '
    '--vocab_size=32000 '
    '--model_type=bpe '
    '--character_coverage=1.0 '
)
