import unittest

from utils.evaluation_schedule import should_evaluate, should_run_final_evaluation


class EvaluationScheduleTests(unittest.TestCase):
    def test_evaluates_on_exact_interval(self):
        self.assertTrue(should_evaluate(step=800, eval_freq=800))

    def test_skips_non_interval_step(self):
        self.assertFalse(should_evaluate(step=799, eval_freq=800))

    def test_skips_zero_step(self):
        self.assertFalse(should_evaluate(step=0, eval_freq=800))

    def test_non_positive_frequency_disables_periodic_evaluation(self):
        self.assertFalse(should_evaluate(step=800, eval_freq=0))
        self.assertFalse(should_evaluate(step=800, eval_freq=-1))

    def test_final_evaluation_runs_when_last_step_was_not_evaluated(self):
        self.assertTrue(
            should_run_final_evaluation(current_step=801, last_eval_step=800))

    def test_final_evaluation_skips_duplicate_step(self):
        self.assertFalse(
            should_run_final_evaluation(current_step=800, last_eval_step=800))

    def test_final_evaluation_skips_empty_training(self):
        self.assertFalse(
            should_run_final_evaluation(current_step=0, last_eval_step=None))


if __name__ == '__main__':
    unittest.main()
