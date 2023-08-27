import pytest
from iml import IML
import torch

@pytest.fixture
def setup_iml():
    src_x=16
    src_y=2
    tgt_size=4

    d_src = (src_x,src_y)
    ctrl = torch.rand(d_src)
    z = torch.zeros(tgt_size)
    iml = IML(d_src, embed='ProjectAndSort')

    def iml_map():
        while(len(iml.pairs) < 32):
            src = torch.rand(d_src)
            tgt = z + torch.randn(tgt_size)*2
            iml.add(src, tgt)
    iml_map()

    return iml, ctrl, z, tgt_size

def test_project_and_sort(setup_iml):
    iml, ctrl, z, tgt_size = setup_iml
    _z = torch.zeros(tgt_size)
    _z[:] = torch.from_numpy(iml.map(ctrl, k=5))
    indices = torch.randperm(ctrl.shape[0])

    def update_pos():
        nonlocal ctrl, indices, z
        indices = torch.randperm(ctrl.shape[0])
        ctrl = ctrl[indices]
        z[:] = torch.from_numpy(iml.map(ctrl, k=5))

    for _ in range(32):
        update_pos()
        assert torch.equal(z, _z), f"Expected z to remain constant, but found difference: {z-_z}"