# Native MEDS predicates: prototype benchmarks

**Prototype, not ready to merge.** See the pull request for the design, results and open questions.

Build the optional extension (needs a C++17 compiler), then run from the repository root:

```bash
pip install -e .   # builds aces._native when a compiler is available
python -c "from aces import native; print(native.available())"

# Plain predicates: identical outputs and timings, every 6th shard
python benchmarks/native/bench_predicates.py /path/to/MEDS/data 6 sample_configs/*.yaml

# End to end (get_predicates_df + query) over every shard
for mode in dev-seq dev-proc8 native-thr8; do
    python benchmarks/native/bench_extraction.py $mode /path/to/MEDS/data \
        sample_configs/inhospital_mortality.yaml labels-$mode.parquet
done
```
