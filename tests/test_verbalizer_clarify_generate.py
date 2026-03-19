import unittest

from src.inference.verbalizer import Verbalizer


class VerbalizerClarifyGenerateTests(unittest.TestCase):
    def test_verbalize_clarify_time(self):
        v = Verbalizer()
        r = v.verbalize_clarify("When?")
        self.assertTrue(isinstance(r.answer_text, str) and len(r.answer_text) > 0)

    def test_verbalize_generate(self):
        v = Verbalizer()
        r = v.verbalize_generate("Why do you think so?")
        self.assertIn("无法", r.answer_text)


if __name__ == "__main__":
    unittest.main()
