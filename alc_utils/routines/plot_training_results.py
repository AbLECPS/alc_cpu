#!/usr/bin/env python2
# Plots supervised learning training results when using keras.

import sys
import os
import json
import matplotlib.pyplot as plt
import matplotlib


def load_data(foldername):
    history = {}
    path1 = os.path.join(foldername, 'history.json')
    if os.path.exists(path1):
        with open(path1) as f:
            history = json.load(f)
        return history

    path_alt = os.path.join(foldername, 'training_results.json')
    if os.path.exists(path_alt):
        with open(path_alt) as f:
            history = json.load(f)
        return history
    return history


# main method that is invoked with folder path
def plot(foldername):
    history = load_data(foldername)
    if len(history.keys()) == 0:
        return
    matplotlib.rcParams['figure.figsize'] = [10, 10]
    if ('acc' in history.keys() or 'val' in history.keys()):
        plt.figure()
        if ('acc' in history.keys()):
            plt.plot(history['acc'], label='Train')
        if 'val_acc' in history.keys():
            plt.plot(history['val_acc'], label='Test')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    if 'val_loss' in history.keys():
        plt.figure()
        plt.plot(history['loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
