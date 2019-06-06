from collections import defaultdict, OrderedDict
import os
import plotly.offline as py
import plotly.tools as pytools
import plotly.graph_objs as go

from mano_train.exputils import logutils


class Monitor:
    def __init__(self, checkpoint, hosting_folder=None):
        self.checkpoint = checkpoint
        self.train_path = os.path.join(self.checkpoint, "train.txt")
        self.val_path = os.path.join(self.checkpoint, "val.txt")
        logutils.create_log_file(self.train_path)
        logutils.create_log_file(self.val_path)

        self.hosting_folder = hosting_folder
        os.makedirs(self.hosting_folder, exist_ok=True)
        self.metrics = Metrics(checkpoint, hosting_folder=self.hosting_folder)

    def log_train(self, epoch, errors):
        logutils.log_errors(epoch, errors, self.train_path)

    def log_val(self, epoch, errors):
        logutils.log_errors(epoch, errors, self.val_path)


class Metrics:
    def __init__(self, checkpoint, hosting_folder=None):
        self.checkpoint = checkpoint
        self.hosting_folder = hosting_folder
        self.evolution = defaultdict(lambda: defaultdict(OrderedDict))

    def save_metrics(self, epoch, metric_dict, val=False):
        for loss_name, loss_dict in metric_dict.items():
            for split_name, val in loss_dict.items():
                self.evolution[loss_name][split_name][epoch] = val

    def plot_metrics(self):
        """For plotting"""
        metric_traces = defaultdict(list)
        for loss_name, loss_dict in self.evolution.items():
            for split_name, vals in loss_dict.items():
                trace = go.Scatter(
                    x=list(vals.keys()),
                    y=list(vals.values()),
                    mode="lines",
                    name=split_name,
                )
                metric_traces[loss_name].append(trace)

        metric_names = list(metric_traces.keys())
        fig = pytools.make_subplots(
            rows=1, cols=len(metric_traces), subplot_titles=tuple(metric_names)
        )

        for metric_idx, metric_name in enumerate(metric_names):
            traces = metric_traces[metric_name]
            for trace in traces:
                fig.append_trace(trace, 1, metric_idx + 1)
        plotly_path = os.path.join(self.checkpoint, "plotly.html")
        py.plot(fig, filename=plotly_path, auto_open=False)
        if self.hosting_folder is not None:
            hosted_plotly_path = os.path.join(self.hosting_folder, "plotly.html")
            py.plot(fig, filename=hosted_plotly_path, auto_open=False)
