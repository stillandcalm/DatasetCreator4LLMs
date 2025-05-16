import sentencepiece as spm

# Load your trained model
sp = spm.SentencePieceProcessor(model_file="llama8b_tokenizer.model")

# Get total number of pieces (tokens)
vocab_size = sp.get_piece_size()
print(f"SentencePiece vocab size: {vocab_size}")

