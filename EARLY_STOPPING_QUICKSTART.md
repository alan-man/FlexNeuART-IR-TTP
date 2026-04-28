# Quick Start: Early Stopping for StackOverflow Tuning

## TL;DR - Compare N=15 Against BEST Previous Configuration!

Add these 3 lines to each experiment command in your notebook:

```python
'-early_stop_enabled', '1',
'-early_stop_threshold', '0',
'-early_stop_at_n', '15'
```

## What's Special About This Setup?

The system now:
- Evaluates N=15 **first** (reordered candidate list)
- **Scans ALL previous hyperparameter configurations** in the same tuning folder
- Finds the **HIGHEST MAP@N=15** among all previous runs
- Compares new N=15 to that best result from ANY previous config
- **Decision**:
  - If **new N=15 > best previous N=15** → Continue full evaluation (all N values)
  - If **new N=15 ≤ best previous N=15** → Stop immediately (save 90% of time!)

## Perfect For Hyperparameter Tuning

Structure:
```
demo/collections/stackoverflow_all/results/dev1/tuning/bm25tune_text_text/
├── bm25tune_k1=0.4_b=0.3/
│   └── rep/out_15.rep (MAP = 0.4400)
├── bm25tune_k1=0.6_b=0.3/
│   └── rep/out_15.rep (MAP = 0.4450) ← BEST!
├── bm25tune_k1=0.8_b=0.3/
│   └── rep/out_15.rep (MAP = 0.4420)
└── bm25tune_k1=1.2_b=0.3/  ← NEW RUN
    └── (running...)
```

When running the new hyperparameter config:
1. Evaluates N=15
2. Scans the 3 previous folders
3. Finds best N=15 = 0.4450
4. Compares new N=15 to 0.4450
5. **If new ≤ 0.4450**: Skip this config, move to next!

## Workflow Example

### Run 1: k1=0.4, b=0.3
```
N=15: MAP = 0.4400 (first config, continues with all N)
Result: Full evaluation (N=1,2,3...100)
Time: 10-15 minutes
Stores: 18 result files
```

### Run 2: k1=0.6, b=0.3
```
N=15: MAP = 0.4450
Best previous: 0.4400 (from Run 1)
Comparison: 0.4450 > 0.4400 ✓ IMPROVED!
Result: Full evaluation (N=1,2,3...100)
Time: 10-15 minutes
Stores: 18 result files
```

### Run 3: k1=0.8, b=0.3 (new attempt)
```
N=15: MAP = 0.4420
Best previous: 0.4450 (from Run 2, which was best)
Comparison: 0.4420 ≤ 0.4450 ✗ NOT BETTER!
Early stopping triggered!
Result: STOPS HERE, only N=15 evaluated
Time: ~1 minute ⚡
Stores: 1 result file
Decision: This config is worse, skip full eval
```

### Run 4: k1=1.2, b=0.3 (another attempt)
```
N=15: MAP = 0.4520
Best previous: 0.4450 (Run 2's best still the reference)
Comparison: 0.4520 > 0.4450 ✓ NEW BEST!
Result: Full evaluation (N=1,2,3...100)
Time: 10-15 minutes
Stores: 18 result files
Decision: This config beats everything, explore fully
```

## Your Notebook Integration

### Before
```python
cmd = [
    'bash', './exper/run_experiments.sh',
    COLLECT_NAME,
    bm25_desc,
    '-test_part', TEST_PART,
    '-train_part', TRAIN_PART,
    '-train_cand_qty', str(TRAIN_CAND_QTY)
]
```

### After (With best-across-all-configs early stopping)
```python
cmd = [
    'bash', './exper/run_experiments.sh',
    COLLECT_NAME,
    bm25_desc,
    '-test_part', TEST_PART,
    '-train_part', TRAIN_PART,
    '-train_cand_qty', str(TRAIN_CAND_QTY),
    '-early_stop_enabled', '1',           # Enable scanning
    '-early_stop_threshold', '0',         # Strict: only continue if new is better
    '-early_stop_at_n', '15'              # Check at N=15
]
```

## What the System Actually Does

**At N=15 evaluation time:**
1. **Scan phase**: Look in all subdirectories of the tuning folder
2. **Extract phase**: Read `out_15.rep` from each previous config
3. **Find max**: Determine highest MAP@N=15 across all previous runs
4. **Compare phase**: New N=15 vs that best previous
5. **Decision**: 
   - Better? → Show message, continue with all N
   - Not better? → Show message, stop early

**Output you'll see:**
```
Best previous N=15 MAP across all configurations: 0.4450
MAP at N=15: 0.4420 (NOT better than previous best 0.4450)
Early stopping: skipping remaining candidate quantities
```

## Tuning Parameters

### Strict (Threshold = 0%)
```python
'-early_stop_threshold', '0'
```
Only continue if `new_MAP > best_previous_MAP` (exactly greater)

### Lenient (Threshold = 0.5%)
```python
'-early_stop_threshold', '0.5'
```
Only continue if `new_MAP > best_previous_MAP * 1.005` (allow 0.5% tolerance)

### Require Improvement (Threshold = -0.5%)
```python
'-early_stop_threshold', '-0.5'
```
Only continue if `new_MAP > best_previous_MAP * 1.005` (must improve by 0.5%)

## FAQ

**Q: Does it really scan all folders?**
A: Yes! It looks in every subdirectory within the tuning folder (the parent of the current config) for `out_15.rep` files.

**Q: What if there are no previous runs?**
A: It shows "No previous N=15 results found (first hyperparameter run)" and continues normally with full evaluation.

**Q: Can I disable it?**
A: Yes: `'-early_stop_enabled', '0'`

**Q: How much time does this save?**
A: If early stopping triggers: ~90% (15 min → 1 min per failed config)

**Q: What's the best strategy?**
A: Run your configs with early stopping enabled. Failed attempts stop immediately, winning configs get fully evaluated. At the end, you've found your best config!

## Implementation Details

The system:
1. Added `findBestMapN15FromPreviousRuns()` function that:
   - Takes current experiment directory
   - Gets parent (tuning) directory
   - Scans ALL subdirs for `rep/out_15.rep`
   - Extracts MAP from each
   - Returns the maximum

2. Uses this in early stopping logic to compare against best previous

3. No special file management needed - just reads existing `out_15.rep` files from siblings

## Full Reference

For more details:
- `N15_FIRST_STRATEGY_EXPLAINED.md` - Detailed explanation
- `EARLY_STOPPING_GUIDE.md` - Complete documentation
