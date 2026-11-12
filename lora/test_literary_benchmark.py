import unittest

from build_literary_benchmark import make_questions
from common import normalize_phrase


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.book = {"name":"fixture", "text":"He walked into the garden. Birds sang loudly. She carried a basket home. Rain began to fall. The lamp glowed brightly. They crossed the bridge.",
                     "events":[{"idx":0,"start":3,"end":9,"text":"walked"},{"idx":1,"start":125,"end":132,"text":"crossed"}]}

    def test_valid_choices_and_gold_alignment(self):
        for strategy in ("random", "hard"):
            for q in make_questions(self.book,"test",5,strategy):
                self.assertEqual(len(q["choices"]),4)
                self.assertEqual(len({normalize_phrase(x) for x in q["choices"]}),4)
                self.assertEqual(q["choices"][q["answer_index"]],q["gold_span"])
                self.assertIn(q["gold_span"],q["passage"])
                for distractor in q["distractors"]: self.assertNotEqual(normalize_phrase(distractor),normalize_phrase(q["gold_span"]))

    def test_reproducible(self):
        self.assertEqual(make_questions(self.book,"test",42,"random"),make_questions(self.book,"test",42,"random"))


if __name__ == "__main__": unittest.main()
