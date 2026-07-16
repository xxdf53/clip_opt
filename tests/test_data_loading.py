import unittest

from utils.data_loading import should_drop_last_batch


class DropLastPolicyTests(unittest.TestCase):
    def test_training_drops_last_batch_by_default(self):
        self.assertTrue(should_drop_last_batch(True, False))

    def test_training_keeps_last_batch_when_requested(self):
        self.assertFalse(should_drop_last_batch(True, True))

    def test_evaluation_always_keeps_last_batch(self):
        self.assertFalse(should_drop_last_batch(False, False))
        self.assertFalse(should_drop_last_batch(False, True))


if __name__ == '__main__':
    unittest.main()
