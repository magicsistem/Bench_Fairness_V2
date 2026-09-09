import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mskcc_icc_d61 import _scope, icc_estimates  # noqa: E402


def legacy_icc_absolute(reference, estimate):
    values = np.column_stack((reference, estimate)); n, k = values.shape
    grand = values.mean(); row_mean, col_mean = values.mean(1), values.mean(0)
    msr = k*np.sum((row_mean-grand)**2)/(n-1); msc = n*np.sum((col_mean-grand)**2)/(k-1)
    mse = np.sum((values-row_mean[:, None]-col_mean[None, :]+grand)**2)/((n-1)*(k-1))
    return float((msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n))


class D61ICCTest(unittest.TestCase):
    def test_identity_is_one(self):
        values = np.arange(1.0, 9.0)
        result = icc_estimates(values, values)
        for key in result:
            self.assertAlmostEqual(result[key]["estimate"], 1.0, places=12)

    def test_constant_offset_preserves_consistency_not_absolute_agreement(self):
        reference = np.arange(10.0)
        result = icc_estimates(reference + 8.0, reference)
        self.assertAlmostEqual(result["icc_3_1_consistency"]["estimate"], 1.0, places=12)
        self.assertLess(result["icc_2_1_absolute_agreement"]["estimate"], 0.5)

    def test_high_noise_reduces_absolute_and_consistency(self):
        reference = np.linspace(-10.0, 10.0, 30)
        clean = icc_estimates(reference + np.sin(reference) * 0.01, reference)
        noisy = icc_estimates(reference + np.random.default_rng(7).normal(0, 25, reference.size), reference)
        for key in ("icc_2_1_absolute_agreement", "icc_3_1_consistency"):
            self.assertLess(noisy[key]["estimate"], clean[key]["estimate"])

    def test_degenerate_input_is_na_with_reason(self):
        result = icc_estimates(np.ones(5), np.ones(5))
        for value in result.values():
            self.assertIsNone(value["estimate"])
            self.assertEqual(value["reason"], "zero_between_target_variance")

    def test_patient_bootstrap_keeps_all_pairs_and_counts_invalid(self):
        records = [
            {"patient_id": "P1", "estimate": 1.0, "reference": 1.1},
            {"patient_id": "P1", "estimate": 2.0, "reference": 2.1},
            {"patient_id": "P2", "estimate": 3.0, "reference": 3.2},
        ]
        result = _scope(records, np.random.default_rng(3), 100)
        self.assertEqual((result["n_images"], result["n_patients"], result["n_effective_pairs"]), (3, 2, 3))
        for key in ("icc_1_1_oneway", "icc_2_1_absolute_agreement", "icc_3_1_consistency"):
            self.assertEqual(result[key]["bootstrap_valid"] + result[key]["bootstrap_invalid"], 100)

    def test_icc2_is_algebraically_identical_to_historical_function(self):
        generator = np.random.default_rng(11)
        reference = generator.normal(size=50); pipeline = reference + generator.normal(scale=.4, size=50)
        observed = icc_estimates(pipeline, reference)["icc_2_1_absolute_agreement"]["estimate"]
        self.assertAlmostEqual(observed, legacy_icc_absolute(reference, pipeline), places=15)

    def test_pingouin_0_5_5_reference_fixture(self):
        # Independently generated with pingouin.intraclass_corr: ICC1, ICC2, ICC3.
        pipeline = np.array([10., 12., 13., 16., 20., 22.])
        colorimeter = np.array([11., 11.5, 15., 18., 19., 25.])
        result = icc_estimates(pipeline, colorimeter)
        expected = {"icc_1_1_oneway": .9362477231329691,
                    "icc_2_1_absolute_agreement": .9367296631059985,
                    "icc_3_1_consistency": .9511096278992158}
        for key, value in expected.items():
            self.assertAlmostEqual(result[key]["estimate"], value, places=12)


if __name__ == "__main__":
    unittest.main()
