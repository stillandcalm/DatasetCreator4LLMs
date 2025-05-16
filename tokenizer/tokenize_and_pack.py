import sentencepiece as spm
from tqdm import tqdm

sp = spm.SentencePieceProcessor(model_file="llama8b_tokenizer.model")
SEQ_LEN = 4096

def pack(lines, out_path):
    buf=[]
    with open(out_path,"w") as fo:
        for ln in tqdm(lines):
            ids = sp.EncodeAsIds(ln)
            buf.extend(ids)
            while len(buf)>=SEQ_LEN:
                chunk, buf = buf[:SEQ_LEN], buf[SEQ_LEN:]
                fo.write(" ".join(map(str, chunk))+"\n")

# load
docs = [l.strip() for l in open("data/scrubbed.txt", encoding="utf-8") if l.strip()]

# split train/test
split = int(0.99*len(docs))
pack(docs[:split], "data/train.sequences")
pack(docs[split:], "data/test.sequences")

