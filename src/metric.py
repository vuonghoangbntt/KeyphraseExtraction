import numpy as np
import logging

def get_f1_score(labels, pred, output_tag):
    from sklearn.metrics import classification_report
    target_names = output_tag.keys()
    report = classification_report(np.array(labels), np.array(pred), target_names=target_names)
    return report