"""The exact resume of a follow-up turn (Service._exact_resume, Service._keep_turn), without a GPU or a model: a byte-level
tokenizer (Basalt's ByteTokenizer, with each token's bytes) and the two Service methods."""
import os
import unittest

from serve.prompt import PromptIds, PromptText
from serve.service import Service
from serve.util.text import ByteTokenizer


class Tok(ByteTokenizer):
    def token_bytes(self, i):
        return self.SPECIALS[i - 256].encode() if i >= 256 else bytes([i])


class Fake:
    """What _exact_resume and _keep_turn use: the tokenizer and the last turn."""
    def __init__(self):
        self.tok, self.last_turn = Tok(), None


P = "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\n<think>\n"
GEN_TEXT = "R1 é.\n</think>\n\nA1 à"
FOLLOW = "<|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n<think>\n"


class ExactResume(unittest.TestCase):
    def turn(self, s, prompt=P, gen_text=GEN_TEXT, n=None):
        p = PromptText(s.tok, prompt)
        p.n = len(s.tok.encode(prompt, parse_special=True)) if n is None else n
        gen = s.tok.encode(gen_text, parse_special=True) + s.tok.encode("<|im_end|>", parse_special=True)
        Service._keep_turn(s, p, gen)
        return gen

    def test_follow_up_goes_as_previous_plus_generated_plus_rest_and_says_the_same_text(self):
        s = Fake()
        gen = self.turn(s)
        text = P + GEN_TEXT + FOLLOW
        ids = Service._exact_resume(s, text, [], "")
        self.assertIsInstance(ids, PromptIds)
        self.assertEqual(ids.text, text)
        self.assertEqual(list(ids), s.tok.encode(P, parse_special=True) + gen[:-1] + s.tok.encode(FOLLOW, parse_special=True))
        self.assertEqual(s.tok.decode(list(ids)), text)

    def test_a_text_that_differs_goes_as_text(self):
        s = Fake()
        self.turn(s)
        self.assertIsNone(Service._exact_resume(s, P + "R1 e.\n</think>\n\nA1 à" + FOLLOW, [], ""))
        self.assertIsNone(Service._exact_resume(s, "<|im_start|>user\nother" + FOLLOW, [], ""))

    def test_literals_forced_opening_and_switch(self):
        s = Fake()
        self.turn(s)
        text = P + GEN_TEXT + FOLLOW
        self.assertIsNone(Service._exact_resume(s, text, [(3, 9)], ""))
        self.assertIsNone(Service._exact_resume(s, text, [], "<tool_call>"))
        os.environ["BASALT_EXACT_RESUME"] = "0"
        try:
            self.assertIsNone(Service._exact_resume(s, text, [], ""))
        finally:
            del os.environ["BASALT_EXACT_RESUME"]
        self.assertIsNotNone(Service._exact_resume(s, text, [], ""))

    def test_a_python_tokenization_without_the_engine_count_does_not_resume(self):
        s = Fake()
        self.turn(s, n=999)
        self.assertIsNone(Service._exact_resume(s, P + GEN_TEXT + FOLLOW, [], ""))

    def test_a_character_split_across_the_last_two_tokens_does_not_resume(self):
        s = Fake()
        # "é" is two bytes: the last generated token is its second byte, the one before its first
        p = PromptText(s.tok, P)
        p.n = len(s.tok.encode(P, parse_special=True))
        Service._keep_turn(s, p, s.tok.encode("R1 é", parse_special=True))
        self.assertIsNone(Service._exact_resume(s, P + "R1 é" + FOLLOW, [], ""))

    def test_a_resumed_turn_is_the_base_of_the_next_one_with_its_own_tokens(self):
        s = Fake()
        gen = self.turn(s)
        t2 = P + GEN_TEXT + FOLLOW
        ids2 = Service._exact_resume(s, t2, [], "")
        gen2 = s.tok.encode("R2.\n</think>\n\nA2", parse_special=True) + s.tok.encode("<|im_end|>", parse_special=True)
        Service._keep_turn(s, ids2, gen2)
        self.assertEqual(s.last_turn["ids"], list(ids2))
        t3 = t2 + "R2.\n</think>\n\nA2" + FOLLOW.replace("Q2", "Q3")
        ids3 = Service._exact_resume(s, t3, [], "")
        self.assertEqual(list(ids3)[:len(ids2) + len(gen2) - 1], list(ids2) + gen2[:-1])
        self.assertEqual(s.tok.decode(list(ids3)), t3)
        p_ids = s.tok.encode(P, parse_special=True)
        self.assertEqual(list(ids3)[:len(p_ids) + len(gen) - 1], p_ids + gen[:-1])

    def test_without_generated_tokens_or_with_a_literal_the_turn_is_not_kept(self):
        s = Fake()
        p = PromptText(s.tok, P)
        Service._keep_turn(s, p, [])
        self.assertIsNone(s.last_turn)
        q = PromptText(s.tok, P, plain=[(0, 3)])
        Service._keep_turn(s, q, [1, 2])
        self.assertIsNone(s.last_turn)


if __name__ == "__main__":
    unittest.main()
