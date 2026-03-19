import unittest

from src.inference.wikidata_retriever import WikidataRetriever


class _MockLinker:
    def link(self, **kwargs):
        class L:
            def __init__(self, qid, label):
                self.qid = qid
                self.label = label

        return [L("Q42", "Douglas Adams")]


class _MockKG:
    def retrieve_1hop(self, **kwargs):
        from src.inference.kg_execute import EvidenceTriple

        return [EvidenceTriple(head="Douglas Adams", relation="author", tail="SomeTail", turn_id=-1, scope="wikidata")]


class WikidataRetrieverMockTests(unittest.TestCase):
    def test_retrieve_mock(self):
        r = WikidataRetriever()
        # monkeypatch internal components
        r.entity_linker = _MockLinker()
        r.kg = _MockKG()

        res, dbg = r.retrieve(head_candidates=["Douglas Adams"], relation_candidate="author", question_text="who?", top_k=1)
        self.assertEqual(res.head_qids, ["Q42"])
        self.assertEqual(res.property_ids, ["P50"])
        self.assertEqual(len(res.evidence), 1)
        self.assertIn("head_qids", dbg)


if __name__ == "__main__":
    unittest.main()
