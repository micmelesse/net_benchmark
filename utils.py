from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_event_acc(log_dir):
    event_acc=EventAccumulator(log_dir)
    event_acc.Reload()
    return event_acc