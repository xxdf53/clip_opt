def should_drop_last_batch(is_train, keep_last_batch=False):
    """Return whether the DataLoader should discard an incomplete final batch."""
    return bool(is_train and not keep_last_batch)
