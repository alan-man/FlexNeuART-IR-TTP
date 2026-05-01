# paths
COLLECT=demo/collections/stackoverflow/results/dev1/tuning/bm25tune_text_text/bm25tune_k1=0.4_b=0.3
TRECDIR="$COLLECT/trec_runs"
REPDIR="$COLLECT/rep"
QREL="demo/collections/stackoverflow/input_data/dev1/qrels.txt"
SCRIPTS="demo/flexneuart_scripts"

runs=(1 2 3 4 5 10 15 20 25 30 35 45 50 60 70 80 90 100)

for n in "${runs[@]}"; do
  echo "== run_$n =="
  orig="$TRECDIR/run_${n}.bz2"
  if [ ! -f "$orig" ]; then
    echo "Missing $orig, skipping"
    continue
  fi

  # backup original
  cp "$orig" "${orig}.orig"

  # create fixed compressed file
  bzcat "$orig" | sed 's/,/./g' | bzip2 > "${orig}.fixed"

  # replace original with fixed (atomic-ish)
  mv "${orig}.fixed" "$orig"

  # create decompressed fixed temp for eval
  tmp="/tmp/run_${n}_fixed"
  bzcat "$orig" | sed 's/,/./g' > "$tmp"

  # run evaluation to regenerate .rep/.tsv
  python3 "$SCRIPTS/exper/eval_output.py" "$QREL" "$tmp" "$REPDIR/out_${n}" "$n"

  echo "Rep file at: $REPDIR/out_${n}.rep"
done