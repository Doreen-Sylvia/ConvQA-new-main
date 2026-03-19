import unittest

from src.inference.controller import Controller
from src.inference.kg_execute import EvidenceTriple


class _DummyExecutor:
    def __init__(self, evidence):
        self._evidence = evidence

    def execute(self, **kwargs):
        return list(self._evidence)


class ControllerTests(unittest.TestCase):
    def test_open_ended_generate(self):
        c = Controller()
        d = c.decide(
            question_text="Why is this important?",
            conv_id="c1",
            memory_executor=_DummyExecutor([]),
        )
        self.assertEqual(d.decision, "GENERATE")

    def test_no_head_clarify(self):
        c = Controller()
        d = c.decide(
            question_text="When?",
            gating_result={"head_candidates": [], "relation_candidate": "publication_year", "use_history": True},
            conv_id="c1",
            memory_executor=_DummyExecutor([]),
        )
        self.assertEqual(d.decision, "ASK_CLARIFY")

    def test_memory_probe_ok_use_memory(self):
        c = Controller()
        ev = [EvidenceTriple(head="H", relation="author", tail="T", turn_id=0, scope="topic")]
        d = c.decide(
            question_text="Who is the author?",
            gating_result={"head_candidates": ["H"], "relation_candidate": "author", "use_history": True},
            conv_id="c1",
            memory_executor=_DummyExecutor(ev),
        )
        self.assertEqual(d.decision, "USE_MEMORY")

    def test_memory_probe_empty_use_wikidata(self):
        c = Controller()
        d = c.decide(
            question_text="Who is the author?",
            gating_result={"head_candidates": ["H"], "relation_candidate": "author", "use_history": True},
            conv_id="c1",
            memory_executor=_DummyExecutor([]),
        )
        self.assertEqual(d.decision, "USE_WIKIDATA")
        self.assertIsInstance(d.wikidata_query, dict)


if __name__ == "__main__":
    unittest.main()
