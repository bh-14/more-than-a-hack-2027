"""Frame-level ball possession for the More than a Hack 2027 challenge.

Layout:
    contract.py   Match (tracking in metres) and Evidence (per-frame scores) -- the shared contract
    geometry.py   derived quantities every evidence module reads: distances, speeds, counts
    evidence/     one module per signal, each returning an Evidence
    decode/       turns summed evidence into H/A/D labels
    io/           loaders for the provided match
    external/     open datasets from other games, and the noise model that makes them look like ATD
    pipeline.py   Match + params -> label codes
"""
