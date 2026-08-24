# Autotuning a linalg.matmul on CPU with upstream MLIR's transform.tune dialect

Proof of concept for autotuning in MLIR using its knob functionality + an external Python program that uses a Gradient Boost Decision Tree (GBDT) and genetic algorithm for the search space, inspired by TVM Ansor.


## Layout

```
payload/matmul_256.mlir              plain linalg.matmul payload, 256x256x256 f32


Test schedules just to find out how MLIR Knobs works (no real usage outside of playing with it)
 
schedules/baseline_fixed_tiles.mlir  payload + transform schedule, literal tile_sizes
schedules/knobs_unresolved.mlir      transform.tune.knob left unresolved -> interpreter failure
schedules/knobs_invalid_selection.mlir  Selected value not in options -> verifier failure
schedules/knobs_resolved_valid.mlir  knobs resolved to valid values -> runs, correct output
schedules/smt_constraint_declaration.mlir  transform.smt.constrain_params

Real schedule used:

schedules/tunable_matmul.mlir        default input payload + unresolved knobs


Python and Bash scripts for the autotuning process:

mlir_schedule.py                     shared knob parse/render/compile/run code
driver/run_one.sh                    JIT reference path: runs one resolved schedule through mlir-opt + mlir-runner (now unused as JIT introduces problem in timing)
driver/compile_schedule.sh           ahead-of-time compiles one resolved schedule to a standalone native executable (used for all timing)
driver/autotune.py                   exhaustive grid search over a schedule's knobs

This generates results such as:

results/grid_search.csv              driver/autotune.py's per-candidate log (generated)
results/best_schedule.mlir           driver/autotune.py's fully-resolved best schedule (generated)



tuner_framework/                     Hardware-agnostic GBDT+GA search library (no MLIR knowledge)
  interfaces.py                        SearchSpace / FeatureExtractor / Evaluator / EvalResult
  tuner.py                             Tuner: the surrogate-guided search loop itself


Concrete example and main PoC:

mlir_tuner/                          MLIR backend for tuner_framework (implements the 3 interfaces)
  backend.py                           KnobSearchSpace / KnobFeatureExtractor / ScheduleEvaluator
  run_gbdt_tuner.py                    CLI entry point, same knobs/flags spirit as driver/autotune.py

results/gbdt_search.csv              run_gbdt_tuner.py's per-candidate log (generated)
results/best_schedule_gbdt.mlir      run_gbdt_tuner.py's fully-resolved best schedule (generated)

custom-tuner/                        Reference only: the original GEMM+simulator GBDT+GA project
                                      tuner_framework/ and mlir_tuner/ are a reusable extraction of its
                                      search algorithm.
```

