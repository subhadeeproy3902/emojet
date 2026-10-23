## What this changes

<!-- One or two sentences. If it fixes an issue, write "Fixes #123". -->

## Type

- [ ] Data (new dataset, lexicon entry, language)
- [ ] Training (notebook builder, loss, evaluation)
- [ ] Test cases (`training/data/eval_sets.json`, seeds)
- [ ] Tooling / scripts
- [ ] Documentation

## If it touches the model or the data

Training runs are long and the easiest mistake is a change that helps one test and quietly hurts another,
so please fill this in:

| Test | Before | After |
|---|---|---|
| real held-out en / hi / bn (hit@6) | | |
| chat cases top-1 / top-3 | | |
| slang probes hit@6 | | |
| lookalikes clean | | |

- Kernel or run link:
- Did anything get **worse**? (If yes, say why it is still worth it.)

> New emoji from the monthly `New emoji watch` PR do not need these numbers — merge the catalog,
> then run a training round so the new outputs are learned.

## Checklist

- [ ] `python3 training/build_notebook.py <commit>` still produces a notebook whose cells compile
- [ ] No data files, model weights or `.env` are committed
- [ ] Datasets added here allow this use, and their licence is recorded next to the loader
