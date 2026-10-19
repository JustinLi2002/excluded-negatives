"""Check that biot5_input.py rebuilds the released BioT5 BindingDB test inputs.

Compares the (input string, label) multiset built from the benchmark's own
random-split test.csv with the instances in task9_bindingdb_drugban_dti_test.json.
An exact match proves the SELFIES conversion, the protein wrapping and the label
mapping are right, so U inputs built the same way are comparable.

Usage: python check_biot5_inputs.py TASK9_JSON RANDOM_TEST_CSV
"""
import csv
import json
import sys
from collections import Counter

from biot5_input import instance_input


def main(task_json, test_csv):
    task = json.load(open(task_json, encoding="utf-8"))
    ref = Counter((i["input"], i["output"][0]) for i in task["Instances"])
    print(f"task file: {len(task['Instances'])} instances, {len(ref)} distinct")
    print("definition:", task["Definition"][0][:80], "...")

    rows = list(csv.DictReader(open(test_csv, newline="")))
    print(f"test.csv: {len(rows)} rows")
    for canonical in (False, True):
        got, failed = Counter(), 0
        for r in rows:
            inp = instance_input(r["SMILES"], r["Protein"], canonical)
            if inp is None:
                failed += 1
                continue
            got[(inp, "Yes." if int(float(r["Y"])) == 1 else "No.")] += 1
        extra = sum((got - ref).values())
        missing = sum((ref - got).values())
        tag = "canonical SMILES" if canonical else "SMILES as released"
        print(f"{tag}: unencodable {failed}, built {sum(got.values())}, "
              f"not in task file {extra}, task-file rows not built {missing}, "
              f"{'IDENTICAL' if got == ref else 'DIFFERENT'}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
