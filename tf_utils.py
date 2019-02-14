from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os
import math
import random
import numpy as np
import pandas as pd
from IPython.display import display, HTML
from tqdm import tqdm
from natsort import natsorted
from protobuf_to_dict import protobuf_to_dict
from tensorflow.python.client import timeline
from functools import lru_cache


def get_event_acc(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    return event_acc


def get_steps(event_acc, scalar_name):
    w_times, steps, vals = zip(*event_acc.Scalars(scalar_name))
    return list(steps)


def get_val(event_acc, scalar_name):
    w_times, step_nums, vals = zip(*event_acc.Scalars(scalar_name))
    return list(vals)


def get_avg_step_time(event_acc, scalar_name):
    w_times, step_nums, vals = zip(*event_acc.Scalars(scalar_name))
    return list(vals)


def tensorboard_smooth(scalars, weight=0.6):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + \
            (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        # Anchor the last smoothed value
        last = smoothed_val

    return smoothed


def create_comparison_plot(scalar_name, A_event_acc, A_label, B_events_acc, B_label):
    A_steps = get_steps(A_event_acc, scalar_name)
    A_vals = tensorboard_smooth(get_val(A_event_acc, scalar_name))

    B_steps = get_steps(B_events_acc, scalar_name)
    B_vals = tensorboard_smooth(get_val(B_events_acc, scalar_name))

    plt.figure()
    plt.gca().set_aspect("auto")
    plt.xlabel("steps")
    plt.ylabel(scalar_name)

    if A_vals == B_vals:
        plt.plot(A_steps, A_vals, color="y", label=A_label+" & "+B_label)
    else:
        plt.plot(A_steps, A_vals, color="r", label=A_label)
        plt.plot(B_steps, B_vals, color="g", label=B_label)

    plt.legend()
    return plt.gcf()


def save_fig(fig, fig_name, dir_name):
    dir_name = "figures/" + dir_name

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.savefig(dir_name + "/" + fig_name.replace('/', '--') +
                ".png", dpi=fig.dpi)


def get_step_gpu_stats(step_metadata):
    step_metadata = protobuf_to_dict(step_metadata)
    step_stats = step_metadata['step_stats']
    dev_stats = step_stats["dev_stats"]
    for d in dev_stats:
        device = d["device"]
        node_stats = d["node_stats"]
        if device == "/job:localhost/replica:0/task:0/device:GPU:0":
            ret = pd.DataFrame(node_stats)
#     print(ret.shape)
    return ret


def get_op_time(node):
    try:
        return node["all_end_rel_micros"]
    except:
        return np.NaN


def get_op_mem(node):
    try:
        return node["memory"][0]["live_bytes"]/1024.0/1024.0
    except:
        return np.NaN


def get_timeline_from_metadata(run_metadata, file_name):
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open(file_name+".json", 'w') as f:
        f.write(chrome_trace)


def collect_nodes(node_name, node_steps):
    ret = []
    for n in node_steps:
        if n["node_name"] == node_name:
            ret.append(n)
#     print(len(ret))
    return ret


def get_common_nodes(A_metrics, B_metrics):
    A_nodes = pd.DataFrame(A_metrics["Node Name"].unique())
    B_nodes = pd.DataFrame(B_metrics["Node Name"].unique())
    common_nodes = pd.merge(A_nodes, B_nodes)
    return common_nodes


def group_nodes(gpu_stats):
    ret = gpu_stats.groupby("node_name").aggregate(lambda x: list(x))
#     print(ret.shape)
    return ret


def extract_metrics(gpu_node_stats):
    metrics = []
    for i, n in gpu_node_stats.iterrows():
        metrics.append([n.name, get_op_mem(n), get_op_time(n)])
    metrics_pd = pd.DataFrame(metrics)
    metrics_pd.columns = ["Node Name", "Memory", "Time"]
    metrics_pd.set_index("Node Name", inplace=True)
#     print(metrics_pd.shape)
    return metrics_pd


def splitup_gpu_stats(gpu_stats_grouped):
    gpu_stats_grouped_encoder = gpu_stats_grouped.filter(
        regex='^dynamic_seq2seq/encoder', axis=0)
    gpu_stats_grouped_attention = gpu_stats_grouped.filter(
        regex='^dynamic_seq2seq/decoder/BahdanauAttention', axis=0)
    gpu_stats_grouped_decoder = gpu_stats_grouped.filter(
        regex='^dynamic_seq2seq/decoder/decoder', axis=0)
#     print(gpu_stats_grouped_encoder.shape,gpu_stats_grouped_attention.shape,gpu_stats_grouped_decoder.shape)
    return gpu_stats_grouped_encoder, gpu_stats_grouped_attention, gpu_stats_grouped_decoder


def scalarify(df):
    return df.applymap(lambda x: x[0] if len(x) == 1 else None)


def filter_nodes(gpu_stats_grouped, regex):
    gpu_stats_grouped_fitlered = gpu_stats_grouped.filter(regex=regex, axis=0)
#     print(gpu_stats_grouped_fitlered.shape)
    return gpu_stats_grouped_fitlered


@lru_cache(maxsize=32)
def process_log(log_dir, regex, max_step=None):
    log_dir = os.path.expanduser(log_dir)

    event_acc = get_event_acc(log_dir)

    if max_step:
        metadata_list = natsorted(event_acc.Tags()["run_metadata"])[:max_step]
    else:
        metadata_list = natsorted(event_acc.Tags()["run_metadata"])

    df_dict = {}
    for step_id in tqdm(metadata_list):
        step_metadata = event_acc.RunMetadata(step_id)
        step_gpu_stats_all = get_step_gpu_stats(step_metadata)
        step_gpu_stats_grouped = group_nodes(step_gpu_stats_all)
        step_gpu_stats_grouped_filtered = filter_nodes(
            step_gpu_stats_grouped, regex)
        df_dict[step_id] = scalarify(step_gpu_stats_grouped_filtered)

    ret = pd.Panel(df_dict)
    return ret


def plot_bar_compare(A_data, A_label, B_data, B_label, filename=None, metric="Time", top_n=-1):
    min_steps = A_data.shape[0] if A_data.shape[0] <= B_data.shape[0] else B_data.shape[0]

    def get_mean_and_std(data):
        ret = (data[:, :, "all_end_rel_micros"]).iloc[0:min_steps, :]
        ret_mean = ret.mean(axis=1)
        ret_std = ret.std(axis=1)
        return ret_mean, ret_std

    A_mean, A_std = get_mean_and_std(A_data)
    B_mean, B_std = get_mean_and_std(B_data)

    data = pd.DataFrame()
    data[A_label+"_mean"] = A_mean
    data[B_label+"_mean"] = B_mean
    data[A_label+"_std"] = A_std
    data[B_label+"_std"] = B_std
    data = data.dropna()
    data["diff"] = data[B_label+"_mean"]-data[A_label+"_mean"]
    data = data.sort_values("diff")
#     data=data.sample(n=top_n,random_state=random.randint(0,2**32 - 1))
    data = data.head(top_n)

    ind = np.arange(len(data.index))
    width = 0.4  # the width of the bars

    plt.figure(figsize=(16, 9))
    ax = plt.gca()
    ax.barh(ind - width/2, data[A_label+"_mean"], width,
            color='Red', label=A_label, xerr=data[A_label+"_std"])
    ax.barh(ind + width/2,  data[B_label+"_mean"], width,
            color='Green', label=B_label, xerr=data[B_label+"_std"])
    ax.legend()
    ax.invert_yaxis()
    plt.yticks(ind, data.index.tolist(), rotation='horizontal', fontsize=10)
    plt.title(u"Time in {}s".format(u"\u03BC"))
    plt.tight_layout()

    return (plt.gcf(), data)
