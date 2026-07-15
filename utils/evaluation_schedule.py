def should_evaluate(step, eval_freq):
    """Return whether periodic evaluation is due after a training step."""
    return step > 0 and eval_freq > 0 and step % eval_freq == 0


def should_run_final_evaluation(current_step, last_eval_step):
    """Return whether training finished on a step that was not evaluated."""
    return current_step > 0 and current_step != last_eval_step
