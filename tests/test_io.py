"""The ATD parser, on a three-frame file written in the documented format."""

import numpy as np

from possession import submission
from possession.contract import AWAY, HOME
from possession.io import bar_ath

META = """<?xml version="1.0"?>
<Root><Teams><Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away</Name></Team></Teams>
<Players><Player id="P1" teamId="T1"/><Player id="P2" teamId="T2"/><Player id="P3" teamId="T1"/>
</Players></Root>
"""

RAW = ("1:0.5,0.5,1.0;NaN,NaN,NaN;0.1,0.2,0.0:0.5,0.5,3.0\n"
       "2:0.5,0.5,1.0;0.6,0.5,2.0;NaN,NaN,NaN:NaN,NaN,NaN\n"
       "3:0.5,0.5,1.0;0.6,0.5,2.0;0.1,0.2,0.0:0.6,0.5,3.0\n")


def test_slot_teams(tmp_path):
    path = tmp_path / "meta.xml"
    path.write_text(META)
    assert bar_ath.slot_teams(str(path)).tolist() == [HOME, AWAY, HOME]


def test_parse_raw(tmp_path):
    path = tmp_path / "raw.txt"
    path.write_text(RAW)
    players, ball = bar_ath.parse_raw(str(path), 3, 3, (100.0, 50.0))
    assert np.allclose(players[0, 0], (50.0, 25.0))
    assert np.isnan(players[0, 1]).all()
    assert np.isnan(ball[1]).all()
    assert np.allclose(ball[2], (60.0, 25.0))


def test_submission_round_trip(tmp_path):
    codes = np.array([3, 0, 1, 2, 3], dtype=np.int8)
    path = submission.write(str(tmp_path / "x.csv"), codes)
    assert submission.validate(path, 5) is None
    assert open(path).read() == "frame,label\n0,X\n1,H\n2,A\n3,D\n4,X\n"
