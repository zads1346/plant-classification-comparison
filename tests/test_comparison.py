import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from compare import CLASSES, FIELDS, build_report, compute_metrics, read_predictions, write_report
from from_oof import convert


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.report = build_report(ROOT / "configs/demo.json")

    def test_demo_counts_and_metrics(self):
        a, b = [m["metrics"] for m in self.report["models"]]
        self.assertEqual((a["n"], a["correct"], a["errors"]), (25, 15, 10))
        self.assertEqual((b["n"], b["correct"], b["errors"]), (25, 18, 7))
        self.assertEqual(a["accuracy"], .60)
        self.assertEqual(b["accuracy"], .72)
        for m in (a, b):
            self.assertEqual(m["micro_f1"], m["accuracy"])
            self.assertEqual(sum(map(sum, m["confusion_matrix"])), 25)
            self.assertEqual(sum(x["support"] for x in m["per_class"]), 25)

    def test_perfect_and_missing_classes(self):
        rows = [dict(true_label=c, pred_label=c) for c in CLASSES]
        self.assertEqual(compute_metrics(rows)["macro_f1"], 1)
        m = compute_metrics(rows[:1])
        self.assertEqual(m["macro_f1"], .2)
        self.assertEqual(m["accuracy"], 1)

    def write_csv(self, path, rows):
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_duplicate_ids_rejected(self):
        rows = self.report["models"][0]["rows"]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.csv"
            self.write_csv(path, rows+[rows[0]])
            with self.assertRaisesRegex(ValueError, "duplicate ID"):
                read_predictions(path, "synthetic")

    def test_wrong_kind_rejected(self):
        with self.assertRaisesRegex(ValueError, "source_kind"):
            read_predictions(ROOT / "examples/demo_a.csv", "validation")

    def test_unknown_label_rejected(self):
        rows = [dict(r) for r in self.report["models"][0]["rows"]]
        rows[0]["pred_label"] = "unknown"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.csv"
            self.write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "unknown class"):
                read_predictions(path, "synthetic")

    def test_input_order_is_not_alignment(self):
        rows = self.report["models"][0]["rows"]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "reversed.csv"
            self.write_csv(path, list(reversed(rows)))
            self.assertEqual(read_predictions(path, "synthetic"), rows)

    def test_mismatched_fold_labels_ids_and_split_rejected(self):
        for field, value in [("fold", "different"), ("true_label", CLASSES[4]), ("ID", "different"), ("split_id", "different")]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                a = [dict(r) for r in self.report["models"][0]["rows"]]
                b = [dict(r) for r in self.report["models"][1]["rows"]]
                if field == "split_id":
                    for r in b: r[field] = value
                else:
                    b[0][field] = value
                self.write_csv(root/"a.csv", a)
                self.write_csv(root/"b.csv", b)
                (root/"cfg.json").write_text(json.dumps(dict(source_kind="synthetic", models=[dict(name="A", path="a.csv"), dict(name="B", path="b.csv")])))
                with self.assertRaisesRegex(ValueError, "exactly the same"):
                    build_report(root/"cfg.json")

    def test_outputs_and_html_safety(self):
        with tempfile.TemporaryDirectory() as d:
            report = self.report
            report["models"][0]["name"] = '</script><script>alert(1)</script>'
            path = write_report(report, Path(d))
            content = path.read_text(encoding="utf-8")
            self.assertNotIn('</script><script>alert(1)', content)
            self.assertIn('\\u003c/script>', content)
            self.assertNotIn('__REPORT_JSON__', content)
            with (Path(d)/"errors.csv").open(encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 17)
            self.assertEqual(len(list(Path(d).glob('confusion_*.csv'))), 2)

    def test_oof_adapter_filters_uncovered_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            input_path, output = root/"oof.csv", root/"converted.csv"
            input_path.write_text('train_relpath,category,predicted_category,fold,covered\na.png,Black-grass,Black-grass,0,True\nb.png,Common wheat,,1,False\n', encoding="utf-8")
            n, split_id = convert(input_path, output)
            self.assertEqual(n, 1)
            rows = read_predictions(output, "validation")
            self.assertEqual(rows[0]["ID"], "a.png")
            self.assertTrue(split_id.startswith("split-"))

    def test_submission_without_ground_truth_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"submission.csv"
            p.write_text('ID,Category\na.png,Black-grass\n')
            with self.assertRaisesRegex(ValueError, "required columns"):
                read_predictions(p, "validation")


if __name__ == "__main__":
    unittest.main()
