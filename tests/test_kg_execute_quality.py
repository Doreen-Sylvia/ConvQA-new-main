import unittest

from src.inference.kg_execute import EvidenceTriple, KGExecutor


class KGExecuteQualityTests(unittest.TestCase):
    def test_tail_type_inference(self):
        self.assertEqual(KGExecutor._tail_type("YEAR::1999"), "YEAR")
        self.assertEqual(KGExecutor._tail_type("1999"), "YEAR")
        self.assertEqual(KGExecutor._tail_type("COUNT::12"), "COUNT")
        self.assertEqual(KGExecutor._tail_type("12"), "COUNT")
        self.assertEqual(KGExecutor._tail_type("BOOL::true"), "BOOL")
        self.assertEqual(KGExecutor._tail_type("true"), "BOOL")
        self.assertEqual(KGExecutor._tail_type("SomeEntity"), "ENTITY")

    def test_sort_key_prefers_type_match_on_tie(self):
        # Arrange: same turn/head rank, different tail types
        head_rank = {"H": 0}
        ev_year = EvidenceTriple(head="H", relation="publication_year", tail="YEAR::2001", turn_id=5, scope="topic")
        ev_entity = EvidenceTriple(head="H", relation="publication_year", tail="SomeEntity", turn_id=5, scope="topic")

        k_year = KGExecutor._sort_key(ev_year, head_rank, rel_cand="publication_year")
        k_entity = KGExecutor._sort_key(ev_entity, head_rank, rel_cand="publication_year")

        # smaller tuple is ranked earlier; year should win due to higher bonus => more negative third element
        self.assertLess(k_year, k_entity)


if __name__ == "__main__":
    unittest.main()
