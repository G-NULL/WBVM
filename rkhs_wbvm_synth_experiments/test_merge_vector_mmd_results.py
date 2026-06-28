import unittest

from merge_vector_mmd_results import merge_table_rows


def row(dataset: str, method: str) -> dict[str, str]:
    return {
        "dataset": dataset,
        "method": method,
        "nfe": "1" if method != "Flow matching" else "20",
        "w2": "0.1",
        "sliced_w2": "0.05",
        "off_manifold_rate": "0.2",
        "off_threshold": "0.1",
        "sample_seconds_10k": "0.01",
        "sample_count": "10000",
    }


class MergeVectorMMDResultsTests(unittest.TestCase):
    def test_prefixes_routes_and_deduplicates_flow_matching(self) -> None:
        merged = merge_table_rows(
            [row("rings2d", "WBVM-all"), row("rings2d", "WBVM-single"), row("rings2d", "Flow matching")],
            [row("rings2d", "WBVM-all"), row("rings2d", "WBVM-single"), row("rings2d", "Flow matching")],
        )

        methods = [r["method"] for r in merged]

        self.assertEqual(
            methods,
            ["RKHS-WBVM-all", "RKHS-WBVM-single", "vMMD-WBVM-all", "vMMD-WBVM-single", "Flow matching"],
        )


if __name__ == "__main__":
    unittest.main()
