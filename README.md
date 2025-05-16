 High-level walk-through of how each of those key intermediate files gets produced:

  $python scripts/crawl.py   --seeds   seeds.txt   --domains domains.txt   --out     data/raw_html.txt
  extracted.txt
      $python scripts/extract_text.py

      Input: data/raw_html.txt (your raw crawler dump, with URL: / ---ENDDOC--- markers)

      What it does:
        Reads in one raw-HTML document at a time (splitting on the URL: and ---ENDDOC--- markers).
        Feeds the HTML blob to Trafilatura to strip out boilerplate and extract the main textual content.
        Writes each cleaned text chunk back-to-back, separated by ---ENDDOC---.
        Result: a plaintext file where each “document” is just the extracted body text, ready for filtering.

  filtered.txt
        
      $python scripts/filter_data.py
      Input: data/extracted.txt

      What it does:
        Splits on ---ENDDOC--- to recover individual documents.
        Length filter: drops anything under a minimum word count (e.g. 100 tokens).
        Language filter: uses langdetect to drop non-English docs.
        Keyword filter: keeps only docs containing at least one term from your cyber_keywords.txt list (e.g. “malware,” “CVE,” “ransomware,” etc.).
        Writes each passing document as a single block (no markers) to data/filtered.txt.
        Result: security-focused English documents of reasonable length.

  deduped.txt
      $python scripts/dedupe.py
      Input: data/filtered.txt

      What it does:
        Reads each line/document.
        Builds a MinHash signature over its word set (using datasketch.MinHash).
        Uses an LSH index to detect any prior document whose Jaccard similarity ≥ 0.8.
        If no near-duplicate is found, it inserts this doc’s MinHash into the index and writes the doc to data/deduped.txt.
      Result: one copy of each “unique” document, with near-duplicates removed.

  all.txt
      Command:
        # From your project root
        grep -v '^$' data/scrubbed.txt > data/all.txt
        Inputs: data/scrubbed.txt (your PII-scrubbed docs, one per line)

      What it does:
        Strips out any blank lines (so SentencePiece doesn’t see empty sentences).
        Concatenates all remaining lines into one large file.
      Result: a single “corpus” file ready to train your BPE tokenizer.

# then run the tokenizer
python scripts/bpetokenizer.py


Putting it all together:

raw_html.txt
   └── extract_text.py  ──▶ extracted.txt
   └── filter_data.py   ──▶ filtered.txt
   └── dedupe.py        ──▶ deduped.txt
   └── scrub_pii.py     ──▶ scrubbed.txt
grep to all.txt               ──▶ all.txt



############### Get the LLAMA3-8B tokenizer.model file 
huggingface-cli download meta-llama/Meta-Llama-3-8B --include "original/tokenizer.model" --local-dir lion1b
#############Get and process data to tokenized train and test set #############
sh finalizedata.sh

#######Incremental Distributed FineTuning of Llama-3-8B basemodel. Follow the scripts in the other tutorial on github######
sh trainingcommand.sh

###########inferencing###############
#option 1
# With explicit prompt
python scripts/inference_cli.py \
  --model_dir output/ll3-8b-ft \
  --prompt "Explain the MITRE ATT&CK framework." \
  --max_new_tokens 128 \
  --temperature 0.7 \
  --top_k 50 \
  --top_p 0.95
#option 2: commandline

#echo "What are the benefits of zero-trust architecture?" | \
#  python scripts/inference_cli.py \
#    --model_dir output/ll3-8b-ft \
#    --max_new_tokens 150



