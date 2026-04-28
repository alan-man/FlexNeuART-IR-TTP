# Early Stopping for Experiment Tuning

## Overview

Early stopping optimizes hyperparameter tuning by **evaluating N=15 first**, then comparing against the **BEST N=15 result from ALL PREVIOUS configurations** to decide whether to continue evaluating higher N values.

**Key idea**: 
- Evaluate N=15 first
- Scan all sibling hyperparameter directories for their best N=15 results
- If your new N=15 > best previous N=15 → continue with all other N values
- If your new N=15 ≤ best previous N=15 → stop, skip all remaining N values (saves ~90% of time!)

## How It Works

### Directory Structure

```
demo/collections/stackoverflow_all/results/dev1/tuning/bm25tune_text_text/
├── bm25tune_k1=0.4_b=0.3/
│   ├── rep/
│   │   └── out_15.rep (MAP = 0.4400)
│   ├── letor/
│   └── trec_runs/
├── bm25tune_k1=0.6_b=0.3/
│   ├── rep/
│   │   └── out_15.rep (MAP = 0.4450) ← BEST!
│   ├── letor/
│   └── trec_runs/
├── bm25tune_k1=0.8_b=0.3/
│   ├── rep/
│   │   └── out_15.rep (MAP = 0.4420)
│   ├── letor/
│   └── trec_runs/
└── bm25tune_k1=1.2_b=0.3/  ← NEW RUN (current)
    ├── rep/ (being populated)
    ├── letor/
    └── trec_runs/
```

### Execution Flow

1. **Reordered candidate list**: N=15 evaluated first (N=15,1,2,3,4,5,10,20,25...)
2. **Scan phase** (at N=15):
   - Look at parent directory (tuning folder)
   - Find all sibling config directories
   - Read `rep/out_15.rep` from each
   - Extract MAP values
   - Keep the maximum (best)
3. **Comparison phase**:
   - New N=15 evaluated
   - New MAP compared to best previous MAP
   - **If new > best**: Continue → evaluate N=1,2,3,4,5,10,20...
   - **If new ≤ best**: Stop → skip to bzipping (no more evaluations)
4. **Results**: Only result files for evaluated N values are created

### Time Savings Example

**Scenario 1: First hyperparameter run**
- Scan: No previous configs found
- Evaluation: Full (all N values)
- Time: ~10-15 minutes
- Result: 18 `.rep` files (one for each N)

**Scenario 2: Later run (no improvement)**
- Scan: Finds best N=15 = 0.4450 across all previous configs
- New N=15 evaluation: MAP = 0.4420
- Comparison: 0.4420 ≤ 0.4450 → STOP
- Time: ~1 minute (only N=15)
- Result: 1 `.rep` file (only N=15)
- Savings: ~90%!

**Scenario 3: Later run (improved)**
- Scan: Finds best N=15 = 0.4450
- New N=15 evaluation: MAP = 0.4520
- Comparison: 0.4520 > 0.4450 → CONTINUE
- Time: ~10-15 minutes (full evaluation)
- Result: 18 `.rep` files
- Action: New best found, explore fully

## Configuration

### Global Defaults (in `scripts/config.sh`)

```bash
DEFAULT_TEST_CAND_QTY_LIST=15,1,2,3,4,5,10,20,25,30,35,45,50,60,70,80,90,100
DEFAULT_EARLY_STOP_ENABLED=0                # Disabled by default
DEFAULT_EARLY_STOP_THRESHOLD=0              # 0% improvement required
DEFAULT_EARLY_STOP_AT_N=15                  # Always check at N=15
```

- `DEFAULT_TEST_CAND_QTY_LIST`: N=15 is first (intentional for early stopping)
- `DEFAULT_EARLY_STOP_ENABLED`: Set to `1` to enable
- `DEFAULT_EARLY_STOP_THRESHOLD`: Percentage improvement
  - `0`: Stop if new MAP ≤ best (strict)
  - `0.5`: Stop if new MAP < best × 1.005 (lenient)
  - `-0.5`: Require 0.5% improvement to continue
- `DEFAULT_EARLY_STOP_AT_N`: Must stay `15` for this strategy

### Per-Experiment Override

```bash
-early_stop_enabled <0|1>
-early_stop_threshold <percentage>
-early_stop_at_n <15>
```

## Usage Example

```python
cmd = [
    'bash', './exper/run_experiments.sh',
    COLLECT_NAME,
    bm25_desc,
    '-test_part', TEST_PART,
    '-train_part', TRAIN_PART,
    '-train_cand_qty', str(TRAIN_CAND_QTY),
    '-early_stop_enabled', '1',           # Enable scanning all configs
    '-early_stop_threshold', '0',         # Strict: new must beat best
    '-early_stop_at_n', '15'              # Always 15
]
```

## Output Messages

### First Configuration Run (No Previous Configs)
```
============================================================
N=15
============================================================
No previous N=15 results found (first hyperparameter run)
MAP at N=15: 0.4450 (first configuration run, no previous results to compare)
```
→ Continues with full evaluation (N=1,2,3...100)

### Run Not Better Than Best
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
```
→ Stops here, no evaluation of N=1,2,3...100

### Run Better Than Best (New Winner!)
```
============================================================
N=15
============================================================
Best previous N=15 MAP across all configurations: 0.4450
MAP at N=15: 0.4520 (improved from previous best 0.4450, continuing with full evaluation)
```
→ Continues with full evaluation (N=1,2,3...100)

## Implementation Details

### New Function: `findBestMapN15FromPreviousRuns()`

```bash
findBestMapN15FromPreviousRuns "$experDirBase" "$REP_SUBDIR"
```

What it does:
1. Takes current experiment base directory (e.g., `...bm25tune_k1=1.2_b=0.3`)
2. Gets parent directory (e.g., `...bm25tune_text_text`)
3. Iterates through all subdirectories
4. Finds and opens `rep/out_15.rep` in each
5. Extracts MAP values
6. Returns the maximum MAP found
7. If none found: returns empty string

### Early Stopping Logic

```bash
# At N=15 evaluation:
bestPreviousMap = findBestMapN15FromPreviousRuns($experDirBase, $REP_SUBDIR)

# Evaluate N=15
currentMap = extract MAP from new evaluation

# Compare
if new > best × (1 + threshold%):
    echo "improved, continuing"
    # Proceed with N=1,2,3...
else:
    echo "not better, stopping"
    earlyStopTriggered=1
    # Skip to bzipping, don't evaluate N=1,2,3...
```

## Comparison Formula

```
improvement = ((new_map - best_previous_map) / best_previous_map) × 100
proceeds = (new_map > best_previous_map × (1 + threshold / 100))
```

Examples with threshold = 0%:
```
Old Best: 0.4450
Run 1 (new): 0.4450 → (0.4450 > 0.4450 × 1.00) = false → STOPS
              (equal, not greater)

Run 2 (new): 0.4451 → (0.4451 > 0.4450 × 1.00) = true → CONTINUES
              (any improvement)

Run 3 (new): 0.4448 → (0.4448 > 0.4450 × 1.00) = false → STOPS
              (worse)
```

## Important Notes

### The System Scans ALL Sibling Directories

- It doesn't ask which config to compare to
- It automatically finds ALL previous configurations
- It picks the BEST N=15 among them
- This ensures you only continue if you beat the current best overall

### First Run Behavior

When no previous configs exist:
- System shows "No previous N=15 results"
- Continues with full evaluation
- Establishes the baseline

### File Structure Must Be Consistent

Scanning assumes:
- All configs are sibling directories in same parent (tuning) folder
- Results are in `rep/out_15.rep` within each config
- Directory naming follows pattern: `configname_param1=val1_param2=val2/`

### Graceful Degradation

If a previous config's `out_15.rep` is missing:
- That config is skipped
- Scanning continues with others
- Best result still found from remaining configs

## Use Cases

### ✅ Perfect For
- Hyperparameter grid search (many configs to try)
- Quick validation of new parameter ranges
- Budget-conscious tuning (time-limited experiments)
- Decision: Keep only configs better than current best

### ⚠️ Less Ideal For
- Single one-off experiments
- When you need complete N=1-100 data regardless
- When exploring very different parameter spaces

## Disabling Early Stopping

### Specific Run
```python
'-early_stop_enabled', '0'
```

### All Runs (in config.sh)
```bash
DEFAULT_EARLY_STOP_ENABLED=0
```

## Troubleshooting

**Q: Early stopping not triggering?**
- Check: Does `earlyStopEnabled = '1'`?
- Check: Are there previous config directories with `rep/out_15.rep`?
- Check: Is the new N=15 MAP actually lower than or equal to best?
- Check: Look for "Best previous N=15 MAP" in output

**Q: System not finding previous configs?**
- Verify directory structure matches expected pattern
- Check all previous configs have `rep/out_15.rep` files
- Look for error messages in scan phase output

**Q: Need full results despite early stopping?**
- Re-run with `-early_stop_enabled 0`
- This won't trigger early stopping, evaluates all N

## Configuration

### Global Defaults (in `scripts/config.sh`)

```bash
DEFAULT_TEST_CAND_QTY_LIST=15,1,2,3,4,5,10,20,25,30,35,45,50,60,70,80,90,100  # N=15 first!
DEFAULT_EARLY_STOP_ENABLED=0                # Disabled by default
DEFAULT_EARLY_STOP_THRESHOLD=0              # 0% improvement required
DEFAULT_EARLY_STOP_AT_N=15                  # Always check at N=15
```

**Key change**: Candidate list now starts with N=15!

- `DEFAULT_EARLY_STOP_ENABLED`: Set to `1` to enable, `0` to disable
- `DEFAULT_EARLY_STOP_THRESHOLD`: Percentage improvement threshold
  - Use `0` to stop if MAP doesn't improve at all (strict)
  - Use `0.5` to allow 0.5% dip (lenient)
  - Use negative value (e.g., `-0.1`) to require minimum improvement
- `DEFAULT_EARLY_STOP_AT_N`: Must stay `15` for N=15-first strategy

### Per-Experiment Override (command-line)

```bash
-early_stop_enabled <0|1>
-early_stop_threshold <percentage>
-early_stop_at_n <15>
```

## Usage Example

### Enable N=15-First Early Stopping

In your experiment tuning cells:

```python
cmd = [
    'bash', './exper/run_experiments.sh',
    COLLECT_NAME,
    bm25_desc,
    '-test_part', TEST_PART,
    '-train_part', TRAIN_PART,
    '-train_cand_qty', str(TRAIN_CAND_QTY),
    '-early_stop_enabled', '1',           # Enable N=15-first checking
    '-early_stop_threshold', '0',         # Strict: stop if any dip
    '-early_stop_at_n', '15'              # Check at N=15
]
```

### Scenario 1: Strict (Stop on Any Plateau)

```python
cmd = [..., '-early_stop_enabled', '1', '-early_stop_threshold', '0']
```

**Behavior**: 
- Evaluates N=15 first
- If N=15 MAP ≤ previous N=15 MAP → stops
- Otherwise → continues with full evaluation

### Scenario 2: Lenient (Allow Small Dips)

```python
cmd = [..., '-early_stop_enabled', '1', '-early_stop_threshold', '0.5']
```

**Behavior**:
- Evaluates N=15 first  
- If N=15 MAP < (previous N=15 MAP × 0.995) → stops
- Otherwise → continues with full evaluation

## Output Messages

When early stopping is enabled, watch for these messages:

### First Run (No Previous Results)
```
============================================================
N=15
============================================================
MAP at N=15: 0.4523 (first evaluation, no previous result to compare)
```

Result: Continues with full evaluation (N=1,2,3...100)

### Subsequent Run (Previous Results Exist, Improved)
```
============================================================
N=15
============================================================
Previous run MAP at N=15: 0.4450 (backing up for comparison)

MAP at N=15: 0.4523 (improved from previous 0.4450, continuing)
```

Result: Continues with full evaluation

### Subsequent Run (No Improvement)
```
============================================================
N=15
============================================================
Previous run MAP at N=15: 0.4523 (backing up for comparison)

MAP at N=15: 0.4380 (no improvement from previous 0.4523)
Early stopping: skipping remaining candidate quantities
============================================================
Early stopping completed: N=15 did not improve, skipped further evaluations
============================================================
```

Result: Stops immediately, skips N=1,2,3,4,5,10,20,25...

## How Comparison Works

### Detecting Previous Results

Before evaluating N=15:
1. Checks if `results/dev1/.../out_15.rep` exists
2. If yes, backs it up to `out_15.rep.backup`
3. Extracts the MAP value from the backup
4. Runs new evaluation (creates new `out_15.rep`)
5. Compares new MAP to backed-up old MAP

### The Comparison Formula

```
improvement = ((new_map - old_map) / old_map) × 100
proceeds = (improvement >= threshold)
```

Example with threshold = 0%:
```
Old MAP: 0.4500
New MAP: 0.4501
Improvement: 0.02% ≥ 0% → Proceeds (continues)

Old MAP: 0.4500
New MAP: 0.4499
Improvement: -0.02% < 0% → Stops early
```

## Important Notes

### Candidate List Now Starts with N=15

Old list: `1,2,3,4,5,10,15,20,25...`  
New list: `15,1,2,3,4,5,10,20,25...`

This is intentional! N=15 is your early stopping decision point.

### Previous Results Must Match

Comparisons only work if:
- Same collection + test part + experiment type
- Results from same or compatible model setup
- The old `out_15.rep` hasn't been deleted

If any change, early stopping will treat it as first run and evaluate normally.

### Integration with Result Collection

The `get_exper_results.sh` script gracefully handles:
- Missing `.rep` files (from skipped evaluations)
- Backup files (ignored during aggregation)
- Mixed early-stopped and full evaluations

## Disabling Early Stopping

### For a Specific Run
```python
'-early_stop_enabled', '0'
```

### For All Runs (in config.sh)
```bash
DEFAULT_EARLY_STOP_ENABLED=0
```

## Use Cases

### ✅ Perfect For
- Iterative parameter tuning (running many similar experiments)
- Quick validation of new ideas (stop if not promising)
- Budget-conscious experiments (time is limited)

### ⚠️ Less Ideal For
- First-time baseline runs (no comparison available)
- Completely new parameter ranges (no relevant history)
- When you need complete N=1-100 data regardless

## Troubleshooting

### Early stopping not triggering
Check:
1. Is `EARLY_STOP_ENABLED = '1'`?
2. Does an old `out_15.rep` exist in the results directory?
3. Is the new N=15 MAP actually lower than or equal to the old one?

### Missing results at higher N values
This is expected when early stopping triggers! The `.rep` files for N>15 won't be created because evaluation stops.

### Want full results after early stop
Re-run with `-early_stop_enabled 0`.

## Implementation Details

### Modified Files

1. `scripts/config.sh`: 
   - Reordered candidate list to put N=15 first
   - Early stopping configuration parameters

2. `scripts/exper/run_one_experiment.sh`:
   - Added N=15-first evaluation logic
   - Backup old `out_15.rep` before evaluation
   - Compare new vs old MAP after evaluation
   - Stop if no improvement

3. `scripts/report/get_exper_results.sh`:
   - Fail-safe `.rep` file reading

### Key Functions

```bash
extractMapFromReport <report_file>    # Extracts MAP value from .rep file
isMapImprovement <baseline> <current> <threshold_pct>  # Threshold-based comparison
```

### Per-Experiment Override (command-line parameters in `run_one_experiment.sh`)

```bash
-early_stop_enabled <0|1>
-early_stop_threshold <percentage>
-early_stop_at_n <number>
```

## Usage Example

### Enable Early Stopping in Your Notebook

In your experiment tuning cells, modify the `run_experiments.sh` command to include early stopping:

```python
cmd = [
    'bash', './exper/run_experiments.sh',
    COLLECT_NAME,
    bm25_desc,
    '-test_part', TEST_PART,
    '-train_part', TRAIN_PART,
    '-train_cand_qty', str(TRAIN_CAND_QTY),
    '-early_stop_enabled', '1',           # Enable early stopping
    '-early_stop_threshold', '0',         # Stop if MAP doesn't improve at all
    '-early_stop_at_n', '15'              # Start checking after N=15
]
```

### Scenario 1: Basic Usage (Stop on First Plateau)

```python
# Uses 0% improvement threshold - stops immediately after N=15 if MAP drops
cmd = [..., '-early_stop_enabled', '1', '-early_stop_threshold', '0']
```

**Results**: Evaluates N = 1,2,3,4,5,10,15, then stops if MAP@15 < best MAP from lower N values.

### Scenario 2: Lenient (Allow Small Plateaus)

```python
# Uses 0.5% improvement threshold - allows slight dips
cmd = [..., '-early_stop_enabled', '1', '-early_stop_threshold', '0.5']
```

**Results**: Only stops if MAP@15 is 0.5% worse than the best previous MAP.

### Scenario 3: Conservative (Deeper Search)

```python
# Only uses early stopping at N=25 and beyond
cmd = [..., '-early_stop_enabled', '1', '-early_stop_at_n', '25']
```

**Results**: Always evaluates up to N=25, then checks stopping condition.

## Time Savings

With the default candidate list (1,2,3,4,5,10,15,20,25,30,35,45,50,60,70,80,90,100):
- Without early stopping: All 18 candidate quantities are evaluated
- With early stopping: Typically 7-9 evaluations (depends on whether plateaus occur)

**Expected speedup**: 50-60% reduction in evaluation time when early stopping triggers

## Output Messages

When early stopping is enabled, watch for these messages in the experiment output:

```
============================================================
N=15
============================================================
MAP at N=15: 0.4523 (no improvement from best 0.4530 at lower N)
Early stopping: skipping candidates beyond N=15
...
============================================================
Early stopping completed: stopped after reaching N=15 without improvement
============================================================
```

## Disabling Early Stopping

To disable for a specific run (overrides global setting):
```python
cmd = [..., '-early_stop_enabled', '0']
```

## Important Notes

1. **TREC runs still generated**: All runs (N=1-100) are computed upfront by Java. Early stopping only skips the evaluation (trec_eval) phase.

2. **Results handling**: The `get_exper_results.sh` script gracefully handles missing `.rep` files from skipped evaluations.

3. **Best practices**:
   - Use **0% threshold** for most cases (stops on any plateau)
   - Use **positive threshold** (0.5-1.0%) if you want to tolerate small fluctuations
   - Use **conservative N threshold** (15-20) to ensure baseline is well-established
   - Disable early stopping for your final/production runs to ensure full evaluation

4. **Interaction with `get_exper_results.sh`**: If you later need full results, you must re-run the evaluation with early stopping disabled.

## Implementation Details

### Modified Files

- `scripts/config.sh`: Added early stopping configuration parameters
- `scripts/exper/run_one_experiment.sh`:
  - Added early stopping helper functions (`extractMapFromReport`, `isMapImprovement`)
  - Modified evaluation loop with early stopping logic
- `scripts/report/get_exper_results.sh`: Made `.rep` file reading fail-safe for missing files

### Key Functions

```bash
extractMapFromReport <report_file>    # Extracts MAP value from .rep file
isMapImprovement <baseline> <current> <threshold_pct>  # Compares with improvement threshold
```
