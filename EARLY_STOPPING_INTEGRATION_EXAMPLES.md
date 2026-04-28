# Early Stopping Integration Example for StackOverflow Notebook

This document shows how to integrate early stopping into your existing experiment workflows.

## Example 1: BM25 Tuning with Early Stopping

**Before** (original notebook cell):
```python
bm25_desc = f'exper_desc/bm25tune_{BM25_QUERY_FIELD}_{BM25_INDEX_FIELD}.json'
cmd = [
    'bash', './exper/run_experiments.sh',
    COLLECT_NAME,
    bm25_desc,
    '-test_part', TEST_PART,
    '-train_part', TRAIN_PART,
    '-train_cand_qty', str(TRAIN_CAND_QTY)
]
print('Running:', ' '.join(cmd))
res = subprocess.run(cmd, cwd=SCRIPTS_DIR, env=os.environ.copy(), text=True, capture_output=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    raise RuntimeError(f'BM25 tuning experiments failed: {res.returncode}')
```

**After** (with early stopping):
```python
# Add early stopping configuration
EARLY_STOP_ENABLED = True          # Enable early stopping
EARLY_STOP_THRESHOLD = 0           # Stop if MAP doesn't improve at all (strict)
EARLY_STOP_AT_N = 15               # Start checking after N=15

bm25_desc = f'exper_desc/bm25tune_{BM25_QUERY_FIELD}_{BM25_INDEX_FIELD}.json'
cmd = [
    'bash', './exper/run_experiments.sh',
    COLLECT_NAME,
    bm25_desc,
    '-test_part', TEST_PART,
    '-train_part', TRAIN_PART,
    '-train_cand_qty', str(TRAIN_CAND_QTY),
    '-early_stop_enabled', '1' if EARLY_STOP_ENABLED else '0',
    '-early_stop_threshold', str(EARLY_STOP_THRESHOLD),
    '-early_stop_at_n', str(EARLY_STOP_AT_N)
]
print('Running:', ' '.join(cmd))
res = subprocess.run(cmd, cwd=SCRIPTS_DIR, env=os.environ.copy(), text=True, capture_output=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    raise RuntimeError(f'BM25 tuning experiments failed: {res.returncode}')
```

## Example 2: Model1+BM25 Tuning with Lenient Early Stopping

```python
# Allow 0.5% MAP dips (useful if metrics fluctuate slightly)
EARLY_STOP_ENABLED = True
EARLY_STOP_THRESHOLD = 0.5         # Allow up to 0.5% decrease
EARLY_STOP_AT_N = 15

model1_desc = f'exper_desc/model1tune_{MODEL1_QUERY_FIELD}_{MODEL1_INDEX_FIELD}.json'
cmd = [
    'bash', './exper/run_experiments.sh',
    COLLECT_NAME,
    model1_desc,
    '-test_part', TEST_PART,
    '-train_part', TRAIN_PART,
    '-train_cand_qty', str(TRAIN_CAND_QTY),
    '-early_stop_enabled', '1' if EARLY_STOP_ENABLED else '0',
    '-early_stop_threshold', str(EARLY_STOP_THRESHOLD),
    '-early_stop_at_n', str(EARLY_STOP_AT_N)
]
print('Running:', ' '.join(cmd))
res = subprocess.run(cmd, cwd=SCRIPTS_DIR, env=os.environ.copy(), text=True, capture_output=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    raise RuntimeError(f'Model1+BM25 tuning experiments failed: {res.returncode}')
```

## Example 3: Multiple Experiments with Configurable Early Stopping

```python
# Configuration at top of notebook
class ExperimentConfig:
    USE_EARLY_STOPPING = True
    EARLY_STOP_THRESHOLD = 0        # Strict: any improvement required
    EARLY_STOP_AT_N = 15
    
    @property
    def early_stop_args(self):
        """Return early stopping arguments as list"""
        if not self.USE_EARLY_STOPPING:
            return []
        return [
            '-early_stop_enabled', '1',
            '-early_stop_threshold', str(self.EARLY_STOP_THRESHOLD),
            '-early_stop_at_n', str(self.EARLY_STOP_AT_N)
        ]

# Usage in experiment cells
config = ExperimentConfig()

bm25_desc = f'exper_desc/bm25tune_{BM25_QUERY_FIELD}_{BM25_INDEX_FIELD}.json'
cmd = [
    'bash', './exper/run_experiments.sh',
    COLLECT_NAME,
    bm25_desc,
    '-test_part', TEST_PART,
    '-train_part', TRAIN_PART,
    '-train_cand_qty', str(TRAIN_CAND_QTY)
] + config.early_stop_args

print('Running:', ' '.join(cmd))
res = subprocess.run(cmd, cwd=SCRIPTS_DIR, env=os.environ.copy(), text=True, capture_output=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    raise RuntimeError(f'Experiments failed: {res.returncode}')
```

## Monitoring Early Stopping in Output

When enabled, watch for these diagnostic messages:

```
============================================================
N=1
============================================================
Tracked best MAP at N=1: 0.4200

============================================================
N=5
============================================================
Updated best MAP at N=5: 0.4450

============================================================
N=15
============================================================
MAP at N=15: 0.4380 (no improvement from best 0.4450 at lower N)
Early stopping: skipping candidates beyond N=15
```

This tells you:
- Best MAP was at N=5 (0.4450)
- MAP dropped at N=15 (0.4380)
- Evaluation stopped, skipping N=20,25,30, etc.

## Performance Comparison

### Without Early Stopping
- Time to evaluate: ~10-15 minutes (evaluates all N up to 100)
- Results: Complete data for all candidate quantities

### With Early Stopping (assuming plateaus early)
- Time to evaluate: ~3-5 minutes (stops after N=15 if no improvement)
- Results: Complete data up to N=15, no data for N>15

## Testing Early Stopping

To verify early stopping is working:

1. **Add debug logging** to see which N values are evaluated:
```python
# After running experiments, check which result files exist:
import os
result_dir = collect_dir / 'results' / TEST_PART / 'tuning' / 'bm25tune_...'
result_files = sorted([f for f in os.listdir(result_dir) if f.startswith('out_') and f.endswith('.rep')])
print(f"Evaluated candidate quantities: {result_files}")
```

2. **Compare with/without early stopping**:
   - Run once with `USE_EARLY_STOPPING = False`
   - Run again with `USE_EARLY_STOPPING = True`
   - Compare execution time and number of result files

## Troubleshooting

### Early stopping not triggering even though enabled

Check:
1. Is `EARLY_STOP_ENABLED = True`?
2. Are there result files for N > 15? If yes, it means MAP kept improving
3. Check the `exper.log` file for early stopping messages

### Missing result files at higher N values

This is expected when early stopping triggers! The `.rep` files for N > 15 (or whatever threshold you set) won't be created. This doesn't affect result collection - `get_exper_results.sh` handles missing files gracefully.

### Need full results after running with early stopping

Re-run the same experiment with `EARLY_STOP_ENABLED = False` to get complete evaluation across all N values.
