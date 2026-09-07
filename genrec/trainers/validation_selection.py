"""Validation-only checkpoint selection; test metrics cannot update patience."""


class ValidationSelection:
    METRICS = {'recall10': 'Recall@10', 'ndcg10': 'NDCG@10'}

    def __init__(self, metric='recall10'):
        if metric not in self.METRICS:
            raise ValueError(f'Unsupported selection_metric: {metric}')
        self.metric = self.METRICS[metric]
        self.best_value = 0.0
        self.best_epoch = -1
        self.counter = 0
        self.best_valid = {}
        self.best_test = {}

    def update(self, epoch, validation_metrics):
        value = validation_metrics[self.metric]
        if value > self.best_value:
            self.best_value = value
            self.best_epoch = epoch
            self.best_valid = dict(validation_metrics)
            self.best_test = {}
            self.counter = 0
            return True
        self.counter += 1
        return False

    def record_selected_test(self, epoch, test_metrics):
        if epoch != self.best_epoch:
            raise ValueError('Test metrics must belong to the validation-selected epoch')
        self.best_test = dict(test_metrics)

    def results(self):
        return {
            'selection_metric': self.metric,
            'best_epoch': self.best_epoch,
            f'best_valid_{self.metric}': self.best_value,
            **{f'best_valid_{k}': v for k, v in self.best_valid.items()},
            **{f'best_test_{k}': v for k, v in self.best_test.items()},
        }
