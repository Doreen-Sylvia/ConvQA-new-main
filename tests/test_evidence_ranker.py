import unittest

from src.inference.evidence_ranker import EvidenceRanker
from src.inference.kg_execute import EvidenceTriple


class EvidenceRankerTests(unittest.TestCase):
    def test_memory_first(self):
        r = EvidenceRanker()
        cands = [
            EvidenceTriple(head="H", relation="publication_year", tail="YEAR::2001", turn_id=-1, scope="wikidata"),
            EvidenceTriple(head="H", relation="publication_year", tail="YEAR::1999", turn_id=3, scope="topic"),
        ]
        out = r.rank(candidates=cands, relation="publication_year", top_m=1)
        self.assertEqual(out.evidence[0].scope, "topic")

    def test_type_constraint_year(self):
        r = EvidenceRanker()
        cands = [
            EvidenceTriple(head="H", relation="publication_year", tail="SomeEntity", turn_id=5, scope="topic"),
            EvidenceTriple(head="H", relation="publication_year", tail="YEAR::2002", turn_id=1, scope="topic"),
        ]
        out = r.rank(candidates=cands, relation="publication_year", top_m=1)
        self.assertEqual(out.pred_answer_value, "2002")

    def test_type_constraint_count(self):
        r = EvidenceRanker()
        cands = [
            EvidenceTriple(head="H", relation="num_books", tail="7", turn_id=2, scope="topic"),
            EvidenceTriple(head="H", relation="num_books", tail="COUNT::9", turn_id=1, scope="topic"),
        ]
        out = r.rank(candidates=cands, relation="num_books", top_m=1)
        self.assertEqual(out.pred_answer_value, "9")


if __name__ == "__main__":
    unittest.main()
