"""
infer_params.token_embedding_device: an unquantized token embedding table held resident on a GPU instead of in
system RAM. The default placement is unchanged, and combinations the option cannot serve are refused.
"""

import pytest
import torch

from exllamav3.modules import Embedding

from testlib.checkpoint import module_config, write_tensors
from testlib.env import get_test_device

DEV = get_test_device()
KEY = "model.embed_tokens"
VOCAB = 600
HIDDEN = 256


def weight_tensors():
    gen = torch.Generator().manual_seed(1)
    return {f"{KEY}.weight": (torch.randn((VOCAB, HIDDEN), generator = gen) * 0.02).half()}


def make_embedding(tmp_path, tensors = None, **infer_params):
    tensors = tensors if tensors is not None else weight_tensors()
    write_tensors(tensors, str(tmp_path))
    config = module_config(tensors, tmp_path, **infer_params)
    return Embedding(config, KEY, VOCAB, HIDDEN, out_dtype = torch.half), tensors


@pytest.mark.nogpu
def test_default_placement_is_cpu(tmp_path):
    module, _ = make_embedding(tmp_path)
    assert module.caps["prefer_cpu"]


@pytest.mark.nogpu
@pytest.mark.parametrize("requested", ("cuda", "cpu", "mps:0"))
def test_requires_indexed_cuda_device(tmp_path, requested):
    with pytest.raises(ValueError, match = "indexed CUDA device"):
        make_embedding(tmp_path, token_embedding_device = requested)


@pytest.mark.nogpu
def test_refuses_quantized_table(tmp_path):
    tensors = {
        f"{KEY}.trellis": torch.zeros((VOCAB, 1, 16), dtype = torch.int16),
        f"{KEY}.signs": torch.ones((1, 256), dtype = torch.half),
    }
    with pytest.raises(ValueError, match = "resident unquantized"):
        make_embedding(tmp_path, tensors, token_embedding_device = "cuda:0")


@pytest.mark.nogpu
def test_refuses_streamed_table(tmp_path):
    with pytest.raises(ValueError, match = "resident unquantized"):
        make_embedding(tmp_path, token_embedding_device = "cuda:0", embed_stream_from_disk = True)


@pytest.mark.nogpu
def test_refuses_other_load_device(tmp_path):
    module, _ = make_embedding(tmp_path, token_embedding_device = "cuda:0")
    assert not module.caps["prefer_cpu"]
    with pytest.raises(ValueError, match = "token_embedding_device"):
        module.load(torch.device("cpu"))
    with pytest.raises(ValueError, match = "token_embedding_device"):
        module.load(torch.device("cuda:1"))


def test_resident_lookup_on_device(tmp_path):
    module, tensors = make_embedding(tmp_path, token_embedding_device = str(DEV))
    module.load(DEV)
    weight = tensors[f"{KEY}.weight"]
    assert module.embedding.weight.device == DEV
    gen = torch.Generator().manual_seed(2)
    for ids in (torch.tensor([[7]]), torch.randint(0, VOCAB, (3, 1), generator = gen),
                torch.randint(0, VOCAB, (2, 100), generator = gen)):
        x = module.prepare_for_device(ids, {})
        assert x.device == DEV
        out = module.forward(x, {})
        assert out.device == DEV and out.dtype == torch.half
        assert torch.equal(out.cpu(), weight[ids])
