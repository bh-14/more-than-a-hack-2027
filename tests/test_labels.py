import numpy as np

from possession.contract import A, D, H, X
from possession.external.labels import controlled_possession


def codes(text):
    return np.array(["HADX".index(c) for c in text], dtype=np.int8)


def test_short_touch_goes_back_to_the_side_before():
    gt = codes("XHHHHHAAHHHHX")
    out = controlled_possession(gt, [(1, 12)], 3)
    assert out.tolist() == codes("XHHHHHHHHHHHX").tolist()


def test_long_enough_spell_is_a_change():
    gt = codes("HHHHAAAAHHHH")
    assert controlled_possession(gt, [(0, 12)], 3).tolist() == gt.tolist()


def test_stoppage_resets_and_is_kept():
    gt = codes("HHHHDDDAAHHHH")
    out = controlled_possession(gt, [(0, 13)], 3)
    # The A spell right after a restart is not 'after the other side', so it stays.
    assert out.tolist() == gt.tolist()
    assert (out == D).sum() == 3 and (out == A).sum() == 2 and (out == H).sum() == 8
