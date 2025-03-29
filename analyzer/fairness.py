from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from sklearn.metrics import accuracy_score

def evaluate_fairness(y_true, y_pred, sensitive_features):
    metric_frame = MetricFrame(
        metrics={
            'accuracy': accuracy_score,
            'selection_rate': selection_rate
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    return {
        "by_group": metric_frame.by_group.to_dict(),
        "overall": metric_frame.overall.to_dict(),
        "demographic_parity_difference": demographic_parity_difference(y_true, y_pred, sensitive_features)
    }