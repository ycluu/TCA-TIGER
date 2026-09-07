import json
import sys
from pathlib import Path

import gin
import pytest

from genrec.trainers.validation_selection import ValidationSelection


def metrics(recall, ndcg):
    return {'Recall@10': recall, 'NDCG@10': ndcg}


def test_default_preserves_recall_and_strict_improvement():
    s = ValidationSelection()
    assert s.update(0, metrics(.1, .05))
    assert not s.update(1, metrics(.09, .06))
    assert not s.update(2, metrics(.1, .07))
    assert s.best_epoch == 0 and s.counter == 2


def test_ndcg_improvement_saves_despite_recall_decline():
    s = ValidationSelection('ndcg10')
    assert s.update(0, metrics(.1, .05))
    assert s.update(1, metrics(.09, .051))
    assert s.best_epoch == 1 and s.counter == 0


def test_declining_validation_and_improving_test_smoke(tmp_path):
    s = ValidationSelection('ndcg10')
    saved = []
    for epoch, ndcg, test_ndcg in [(0, .050, .040), (1, .049, .045), (2, .051, .041)]:
        improved = s.update(epoch, metrics(.1, ndcg))
        if improved:
            saved.append(epoch)
            s.record_selected_test(epoch, metrics(.2, test_ndcg))
        assert s.counter == (1 if epoch == 1 else 0)
        assert s.best_epoch == (0 if epoch == 1 else epoch)
    assert saved == [0, 2]
    # Exercise the same JSON writer as the trainer.
    from genrec.trainers.trainer_utils import save_run_results
    path = save_run_results(str(tmp_path), 'tiger', 'beauty', 42, s.results())
    result = json.loads(Path(path).read_text())['metrics']
    assert result['selection_metric'] == 'NDCG@10'
    assert result['best_epoch'] == 2
    assert result['best_valid_NDCG@10'] == .051
    assert result['best_test_NDCG@10'] == .041  # Not maximum test .045.


def test_test_metrics_never_reset_patience_or_select_epoch():
    s = ValidationSelection('ndcg10')
    s.update(0, metrics(.1, .05))
    s.record_selected_test(0, metrics(.1, .04))
    s.update(1, metrics(.2, .049))
    with pytest.raises(ValueError, match='validation-selected'):
        s.record_selected_test(1, metrics(1., 1.))
    assert s.counter == 1 and s.best_epoch == 0
    assert s.best_test['NDCG@10'] == .04


@pytest.mark.parametrize('metric', ['NDCG@10', 'ndcg', 'test_ndcg10', 'unsupported'])
def test_unsupported_metric(metric):
    with pytest.raises(ValueError, match='Unsupported'):
        ValidationSelection(metric)


def test_new_config_only_changes_selection_and_output(monkeypatch):
    from genrec.modules.utils import parse_config
    from genrec.trainers.tiger_trainer import train
    prefix = 'config/tiger/amazon/tiger_rqkmeans_tca_full_vocab'

    def read_config(path):
        gin.clear_config()
        monkeypatch.setattr(sys, 'argv', ['audit', path, '--split', 'beauty'])
        parse_config()
        return dict(gin.get_bindings(train))

    try:
        old = read_config(prefix + '.gin')
        new = read_config(prefix + '_ndcg_select.gin')
        assert new.pop('selection_metric') == 'ndcg10'
        new_dir = new.pop('save_dir_root')
        old_dir = old.pop('save_dir_root')
        assert new_dir != old_dir
        assert new_dir.endswith('/tiger_tca_full_vocab_ndcg_select')
        assert old == new  # Includes every bound training hyperparameter/path.
        recorded = json.loads(Path(old_dir, 'results.json').read_text())['config']
        for key, value in old.items():
            if key != 'dataset':
                assert value == recorded[key], key
        assert old['eval_test_every_epoch'] == 2
    finally:
        gin.clear_config()
