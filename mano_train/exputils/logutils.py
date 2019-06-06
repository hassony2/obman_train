from collections import defaultdict
from operator import itemgetter
import os
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def create_log_file(log_path, log_name=""):
    # Make log folder if necessary
    log_folder = os.path.dirname(log_path)
    os.makedirs(log_folder, exist_ok=True)

    # Initialize log files
    with open(log_path, "a") as log_file:
        now = time.strftime("%c")
        log_file.write("==== log {} at {} ====\n".format(log_name, now))


def log_errors(epoch, errors, log_path=None):
    """log_path overrides the destination path of the log
    Args:
        valid(bool): Whether to use the default valid or train log file
            (overriden by log_paht)
        errors(dict): in format {'error_name':error_score, ...}
    """
    now = time.strftime("%c")
    message = "(epoch: {epoch}, time: {t})".format(epoch=epoch, t=now)
    for k, v in errors.items():
        message = message + ",{name}:{err}".format(name=k, err=v)

    # Write log message to correct file
    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")
    return message


def get_logs(path):
    """Processes logs in format "(somehting),loss_1:0.1234,loss_2:0.3"
    """
    logs = defaultdict(list)
    with open(path, "r") as log_file:
        for line in log_file:
            if line[0] != "=":
                prefix, results = line.strip().split(")")
                results = results.split(",")
                epoch = prefix[1:].split(",")[0].split(": ")[1]
                logs["epoch"] = epoch

                for score in results:
                    if ":" in score:
                        score_name, score_value = score.split(":")
                        logs[score_name].append(float(score_value))
    return logs


def process_logs(logs, plot_metric="top1", score_iter=20):
    iter_scores = []
    for score_name, score_values in logs.items():
        assert_message = "index {} out of range for score_values of len {}".format(
            score_iter, len(score_values)
        )
        assert len(score_values) > score_iter, assert_message

        iter_scores.append((score_name, score_values[score_iter]))
    return sorted(iter_scores, key=itemgetter(0))


def plot_logs(logs, score_name="top1", y_max=1, prefix=None, score_type=None):
    """
    Args:
        score_type (str): label for current curve, [valid|train|aggreg]
    """

    # Plot all losses
    scores = logs[score_name]
    if score_type is None:
        label = prefix + ""
    else:
        label = prefix + "_" + score_type.lower()

    plt.plot(scores, label=label)
    plt.title(score_name)
    if score_name == "top1" or score_name == "top1_action":
        # Set maximum for y axis
        plt.minorticks_on()
        x1, x2, _, _ = plt.axis()
        axes = plt.gca()
        axes.yaxis.set_minor_locator(MultipleLocator(0.02))
        plt.axis((x1, x2, 0, y_max))
        plt.grid(b=True, which="minor", color="k", alpha=0.2, linestyle="-")
        plt.grid(b=True, which="major", color="k", linestyle="-")


def display_logs(
    log_file, score_type, score_iter=10, plot_metric="top1", prefix=None, vis=True
):
    """Process logs, prints the results for the given score_iter
    and plots the matching curves
    """
    logs = get_logs(log_file)
    print("Metrics: {}".format(list(logs.keys())))
    iter_scores = process_logs(logs, score_iter=score_iter, plot_metric=plot_metric)
    # Plot all losses
    if vis:
        plot_logs(logs, prefix=prefix, score_name=plot_metric, score_type=score_type)

        print("==== {} scores ====".format(score_type))
        for loss, val in iter_scores:
            print("{val}: {loss}".format(val=val, loss=loss))
