import pytest

pytest.importorskip("torch")
from pytorch3dunet.unet3d.trainer import UNetTrainer  # noqa: E402


class _Dummy:
    def __init__(self, lr=1e-4):
        self.param_groups = [{"lr": lr}]

    def eval(self):
        pass

    train = eval

    def close(self):
        pass


def _trainer(scores, patience):
    t = UNetTrainer.__new__(UNetTrainer)
    t.model, t.optimizer, t.scheduler = _Dummy(), _Dummy(), None
    t.writer, t.tb_drive_writer = _Dummy(), None
    t.num_epochs, t.max_num_epochs = 0, len(scores)
    t.eval_score_higher_is_better, t.best_eval_score = True, float("-inf")
    t.patience = patience
    t.epochs_without_improvement = 0
    it = iter(scores)
    t.train = lambda: None
    t.validate = lambda: next(it)
    t._log_lr = lambda: None
    t._save_checkpoint = lambda is_best: None
    return t


def test_stops_after_patience_epochs_without_improvement():
    t = _trainer([0.1, 0.2, 0.15, 0.18, 0.19, 0.5, 0.6], patience=3)
    t.fit()
    assert t.num_epochs == 5  # best at epoch 2, then 3 epochs without improvement
    assert t.best_eval_score == 0.2


def test_patience_disabled_runs_all_epochs():
    t = _trainer([0.3, 0.2, 0.1, 0.05], patience=None)
    t.fit()
    assert t.num_epochs == 4
