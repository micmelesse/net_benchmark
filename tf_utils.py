from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def get_event_acc(log_dir):
    event_acc=EventAccumulator(log_dir)
    event_acc.Reload()
    return event_acc

def get_steps(event_acc,scalar_name):
    w_times, steps, vals = zip(*event_acc.Scalars(scalar_name))
    return list(steps)

def get_val(event_acc,scalar_name):
    w_times, step_nums, vals = zip(*event_acc.Scalars(scalar_name))
    return list(vals)

def get_avg_step_time(event_acc,scalar_name):
    w_times, step_nums, vals = zip(*event_acc.Scalars(scalar_name))
    return list(vals)

def tensorboard_smooth(scalars, weight=0.6):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def create_comparison_plot(scalar_name,A_event_acc,A_label,B_events_acc,B_label):
    A_steps=get_steps(A_event_acc,scalar_name)
    A_vals=tensorboard_smooth(get_val(A_event_acc,scalar_name))
    
    B_steps=get_steps(B_events_acc,scalar_name)
    B_vals=tensorboard_smooth(get_val(B_events_acc,scalar_name))
    
    plt.figure()
    plt.gca().set_aspect("auto")
    plt.xlabel("steps")
    plt.ylabel(scalar_name)
      
    plt.plot(A_steps,A_vals,color="r",label=A_label)
    plt.plot(B_steps,B_vals,color="g",label=B_label)
    
   
    plt.legend()
    return plt.gcf()