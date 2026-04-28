# Hyperparameter Tuning with Early Stopping: Complete Strategy

## Your Goal (In Your Words)

*"After running N=15, check all previous configurations. My goal in the end is to choose the best configuration with highest MAP for N=15. And if MAP is highest, compute the following N, else early stopping."*

## What We Implemented

Each time you run a **new hyperparameter configuration**:

1. ✅ Evaluate **N=15 FIRST** (not after other N values)
2. ✅ **Scan ALL PREVIOUS configurations** in the same tuning folder
3. ✅ Find the **HIGHEST MAP@N=15** among all previous hyperparameter runs
4. ✅ **Compare new N=15 against that best**:
   - If `new N=15 > best previous N=15` → **Continue** with full evaluation (all N)
   - If `new N=15 ≤ best previous N=15` → **Stop early**, skip unnecessary evaluations

---

## Your Directory Structure

```
demo/collections/stackoverflow_all/results/dev1/tuning/bm25tune_text_text/
│
├─ bm25tune_k1=0.4_b=0.3/            (Configuration A)
│  ├─ rep/
│  │  ├─ out_1.rep
│  │  ├─ out_15.rep  (MAP = 0.4400)
│  │  ├─ out_2.rep
│  │  └─ ...
│  ├─ letor/
│  └─ trec_runs/
│
├─ bm25tune_k1=0.6_b=0.3/            (Configuration B)
│  ├─ rep/
│  │  ├─ out_1.rep
│  │  ├─ out_15.rep  (MAP = 0.4450) ← BEST!
│  │  ├─ out_2.rep
│  │  └─ ...
│  ├─ letor/
│  └─ trec_runs/
│
├─ bm25tune_k1=0.8_b=0.3/            (Configuration C)
│  ├─ rep/
│  │  ├─ out_1.rep
│  │  ├─ out_15.rep  (MAP = 0.4420)
│  │  ├─ out_2.rep
│  │  └─ ...
│  ├─ letor/
│  └─ trec_runs/
│
└─ bm25tune_k1=1.2_b=0.3/            (Configuration D - RUNNING NOW)
   ├─ rep/  (being populated)
   │  ├─ out_1.rep  (to be created)
   │  ├─ out_15.rep  (evaluating NOW)
   │  └─ ...
   ├─ letor/
   └─ trec_runs/
```

---

## How It Works: Step-by-Step

### When You Run a New Configuration

```bash
$ bash run_experiments.sh ... \
  -early_stop_enabled 1 \
  -early_stop_threshold 0 \
  -early_stop_at_n 15
```

### Step 1: Reach N=15 Evaluation

System prints:
```
============================================================
N=15
============================================================
```

### Step 2: SCAN Phase (Find Best Previous)

**System scans the parent directory:**
```
Scanning: bm25tune_text_text/

Looking for all out_15.rep files...
├─ bm25tune_k1=0.4_b=0.3/rep/out_15.rep → MAP = 0.4400
├─ bm25tune_k1=0.6_b=0.3/rep/out_15.rep → MAP = 0.4450 ← BEST!
├─ bm25tune_k1=0.8_b=0.3/rep/out_15.rep → MAP = 0.4420
└─ [skipping self: bm25tune_k1=1.2_b=0.3/]

Result: Best previous N=15 MAP = 0.4450
```

**System prints:**
```
Best previous N=15 MAP across all configurations: 0.4450
```

### Step 3: EVALUATE Phase (New N=15)

**Runs the new evaluation:**
```bash
./exper/eval_output.py $qrels ... rep/out_15
```

**Creates:** `bm25tune_k1=1.2_b=0.3/rep/out_15.rep` with new MAP

**System prints:**
```
MAP at N=15: 0.4480
```

### Step 4: COMPARE Phase (Decision)

```
New N=15 MAP:      0.4480
Best previous MAP: 0.4450
Threshold:         0% (strict)

Comparison: 0.4480 > 0.4450? YES ✓
```

**Decision: CONTINUE ✓**

System prints:
```
improved from previous best 0.4450, continuing with full evaluation
```

Continues evaluating: N=1, N=2, N=3, ..., N=100

---

## Real Example: Four Configurations

### Configuration A (Baseline)

```
Command: run_experiments.sh ... -early_stop_enabled 1

Scan Phase:
  Result: No previous configs found
  
N=15 Evaluation:
  MAP = 0.4400
  
Decision: CONTINUE ✓ (first run, establishes baseline)

Time: 10-15 minutes
Output: out_1.rep, out_2.rep, ..., out_100.rep (18 files)

Status: ✓ Baseline established (MAP@N=15 = 0.4400)
```

### Configuration B (Better)

```
Command: run_experiments.sh ... -early_stop_enabled 1

Scan Phase:
  Finds: Configuration A out_15.rep (MAP = 0.4400)
  Best: 0.4400
  
N=15 Evaluation:
  MAP = 0.4450
  
Comparison: 0.4450 > 0.4400? YES ✓
  
Decision: CONTINUE ✓ (improved, explore fully)

Time: 10-15 minutes
Output: out_1.rep, out_2.rep, ..., out_100.rep (18 files)

Status: ✓ NEW BEST CONFIG (MAP@N=15 = 0.4450)
```

### Configuration C (Not Better)

```
Command: run_experiments.sh ... -early_stop_enabled 1

Scan Phase:
  Finds: Configuration A out_15.rep (MAP = 0.4400)
  Finds: Configuration B out_15.rep (MAP = 0.4450) ← BEST!
  Best: 0.4450
  
Message Printed:
  "Best previous N=15 MAP across all configurations: 0.4450"

N=15 Evaluation:
  MAP = 0.4420
  
Comparison: 0.4420 > 0.4450? NO ✗
  
Message Printed:
  "MAP at N=15: 0.4420 (NOT better than previous best 0.4450)"
  "Early stopping: skipping remaining candidate quantities"

Decision: STOP ✗ (early stopping triggered)

Time: ~1 minute (ONLY N=15 evaluated)
Output: ONLY out_15.rep (1 file)

Status: ✗ Not competitive, skipped. SAVED 90% OF TIME!
```

### Configuration D (Even Better)

```
Command: run_experiments.sh ... -early_stop_enabled 1

Scan Phase:
  Finds: Configs A, B, C's out_15.rep files
  Best from all: 0.4450 (still Configuration B)
  
Message Printed:
  "Best previous N=15 MAP across all configurations: 0.4450"

N=15 Evaluation:
  MAP = 0.4520
  
Comparison: 0.4520 > 0.4450? YES ✓
  
Message Printed:
  "MAP at N=15: 0.4520 (improved from previous best 0.4450, continuing with full evaluation)"

Decision: CONTINUE ✓ (new best found!)

Time: 10-15 minutes
Output: out_1.rep, out_2.rep, ..., out_100.rep (18 files)

Status: ✓ NEW BEST OVERALL! (MAP@N=15 = 0.4520)
         (Next configs will compare against 0.4520)
```

---

## What It Prints to Screen

### First Configuration

```
============================================================
N=15
============================================================
No previous N=15 results found (first hyperparameter run)
MAP at N=15: 0.4400 (first configuration run, no previous results to compare)

[continues with N=1, N=2, ..., N=100]
```

### Subsequent Configuration (Improves)

```
============================================================
N=15
============================================================
Best previous N=15 MAP across all configurations: 0.4450
MAP at N=15: 0.4480 (improved from previous best 0.4450, continuing with full evaluation)

[continues with N=1, N=2, ..., N=100]
```

### Subsequent Configuration (No Improvement)

```
============================================================
N=15
============================================================
Best previous N=15 MAP across all configurations: 0.4450
MAP at N=15: 0.4420 (NOT better than previous best 0.4450)
Early stopping: skipping remaining candidate quantities

============================================================
Early stopping completed: N=15 did not beat best previous, skipped further evaluations
============================================================

[bzipping files - only out_15.rep exists]
```

---

## How to Use in Your Notebook

Simply add these 3 parameters to your experiment commands:

```python
cmd = [
    'bash', './exper/run_experiments.sh',
    COLLECT_NAME,
    bm25_desc,
    '-test_part', TEST_PART,
    '-train_part', TRAIN_PART,
    '-train_cand_qty', str(TRAIN_CAND_QTY),
    # ↓ ADD THESE THREE LINES ↓
    '-early_stop_enabled', '1',           # Enable scanning all configs
    '-early_stop_threshold', '0',         # Strict: new must beat best
    '-early_stop_at_n', '15'              # Check at N=15 only
    # ↑ ADD THESE THREE LINES ↑
]
```

That's it! The rest happens automatically.

---

## Threshold Options

| Value | Meaning | Example |
|-------|---------|---------|
| `'0'` | Only continue if new > best (strictly greater) | 0.4450 → 0.4451 ✓ |
| `'0.5'` | Allow 0.5% margin | 0.4450 → 0.4428 ✓ (0.5% dip OK) |
| `'-0.5'` | Require 0.5% improvement | 0.4450 → 0.4472 ✓ (0.5% gain needed) |

**Recommendation**: Start with `'0'` (strict). Use `'0.5'` only if metrics fluctuate.

---

## Performance Summary

### Single Run Costs

| Type | Time | Result Files | Continuation |
|------|------|--------------|---------------|
| First config | 10-15 min | 18 | Always (establishes baseline) |
| Better than best | 10-15 min | 18 | Yes (explore fully) |
| **Not better than best** | **~1 min** | **1** | **NO (stops early)** |

### Grid Search Example (20 configs)

```
Configs tested: 20

Best case (no improvement):
  1st config: 15 min (baseline)
  2-20: ~1 min each (early stop)
  Total: 15 + 19 = 34 minutes
  
Without early stopping:
  1-20: 15 min each
  Total: 300 minutes
  
TIME SAVED: 266 minutes = ~4.5 hours! ⚡
```

---

## Implementation Details

### New Function Added

```bash
findBestMapN15FromPreviousRuns(experDirBase, repSubdir)
```

This function:
1. Takes current experiment directory
2. Finds parent (tuning) directory
3. Scans ALL sibling directories
4. Reads `rep/out_15.rep` from each
5. Returns the **maximum MAP** found
6. Returns empty if none found (first config)

### Logic Flow

```bash
# At N=15 evaluation:

bestMap = findBestMapN15FromPreviousRuns(...)

if bestMap is empty:
  # First config, print message and continue
  echo "No previous N=15 results found"
  continue with all N
  
else:
  # Previous configs exist
  echo "Best previous N=15 MAP: $bestMap"
  
  evaluate N=15
  currentMap = extract MAP
  
  if currentMap > bestMap × (1 + threshold%):
    echo "improved, continuing"
    proceed with full eval
  else:
    echo "not better, stopping"
    earlyStopTriggered=1
    skip N=1,2,3...
```

---

## Key Advantages

✅ **Automatic scanning**: No manual directory navigation  
✅ **Always against actual best**: Not just "previous run" but BEST overall  
✅ **Time efficient**: Bad configs stop immediately (~1 min vs 15 min)  
✅ **Decision driven**: Only explore configs that beat current best  
✅ **Comprehensive**: Scans ALL previous hyperparameter combinations  
✅ **Fair comparison**: Each config compared fairly against all history

---

## Troubleshooting

**Q: Why didn't early stopping trigger?**
- Verify `-early_stop_enabled '1'` (string, not integer)
- Check that previous config folders have `rep/out_15.rep` files
- Verify new N=15 MAP > best previous MAP

**Q: Why are result files missing for high N?**
- This is expected! Early stopping worked
- Re-run with `-early_stop_enabled 0` if you need complete data

**Q: How do I know which config was best?**
- Look for "improved from previous best" messages
- Check each config's `rep/out_15.rep` file
- Last config with full 18 result files is likely best (unless a later one stopped early)

---

## Your Workflow

1. **Setup**: Add the 3 early stopping parameters to your notebook
2. **Run first config**: Establishes baseline (full evaluation)
3. **Run subsequent configs**: 
   - Fast filter if not better (early stop, ~1 min)
   - Full exploration if better than current best (15 min)
4. **At the end**: Your directory has:
   - Multiple config folders
   - Each with N=15 results (comparable)
   - Best configs with full N=1-100 results
   - You've found your best configuration!

5. **Decision**: Re-run the best config ONE MORE TIME without early stopping (`-early_stop_enabled 0`) to get complete N=1-100 data for publication/final reports

---

## Files Modified

- ✏️ `scripts/config.sh`: Reordered candidate list (N=15 first)
- ✏️ `scripts/exper/run_one_experiment.sh`: Added scanning + comparison logic
- ✏️ `scripts/report/get_exper_results.sh`: Made result reading fail-safe

All changes are backward compatible (early stopping disabled by default).
