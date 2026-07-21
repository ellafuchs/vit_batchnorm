import torch

from vitbn.norm import BatchNorm


def test_preserves_shape():
    assert BatchNorm(192)(torch.randn(4, 197, 192)).shape == (4, 197, 192)


def test_statistics_are_per_channel_pooled_over_images_and_tokens():
    """Standard BatchNorm: one mean/var per channel, pooling everything else."""
    m = BatchNorm(8)
    m.train()
    y = m(torch.randn(16, 20, 8) * 5 + 3)
    flat = y.reshape(-1, 8)
    assert torch.allclose(flat.mean(0), torch.zeros(8), atol=1e-5)
    assert torch.allclose(flat.std(0, unbiased=False), torch.ones(8), atol=1e-4)


def test_has_one_stat_pair_per_channel():
    m = BatchNorm(192)
    assert m.bn.running_mean.shape == (192,)
    assert m.bn.running_var.shape == (192,)


def test_eval_mode_is_batch_independent():
    """The property Hailo needs: at inference the output must not depend on
    what else is in the batch, so batch size 1 equals batch size 8."""
    m = BatchNorm(16)
    m.train()
    m(torch.randn(32, 197, 16))      # populate running stats
    m.eval()
    x = torch.randn(8, 197, 16)
    alone = m(x[:1])
    together = m(x)[:1]
    assert torch.allclose(alone, together, atol=1e-6)
