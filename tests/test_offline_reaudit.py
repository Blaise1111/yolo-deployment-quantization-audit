import numpy as np
import pandas as pd
import pytest

from src.coco_split import write_json
from src.offline_reaudit import operating_row, post_nms_cohorts, timing_pairs, validate_partitions
from src.timing_evidence import load, path_for, publish, validate_order


def evidence():
    return pd.DataFrame([dict(resolution=640,device="cpu",round=1,image_id=19,
                              backend=b,decode_ms=1.,preprocess_ms=2.,inference_ms=t,
                              postprocess_ms=2.,total_ms=t+5,host_threads=4,host_affinity="0,1,2,3")
                         for b,t in (("onnx_fp32",3.),("onnx_int8",6.))])


def test_timing_coverage_and_validity():
    rows=evidence()
    assert timing_pairs(rows).total_ms_paired_delta.tolist()==[3.]
    with pytest.raises(ValueError,match="coverage"):
        timing_pairs(rows.iloc[:1])
    with pytest.raises(ValueError,match="duplicate"):
        timing_pairs(pd.concat([rows,rows]))
    with pytest.raises(ValueError,match="summing"):
        timing_pairs(rows.assign(total_ms=1))


def test_timing_contract_digest_order_and_partial(tmp_path):
    rows=evidence()
    expected=dict(sample_order=[dict(id=19)],jobs=[dict(size=640,backend=b,device="cpu")
                   for b in rows.backend],rounds=1,affinity=[0,1,2,3],intra_threads=4)
    path=tmp_path/"timing.csv"
    rows.to_csv(path,index=False)
    with pytest.raises(ValueError,match="manifest"):
        load(path,expected)
    publish(path,expected)
    assert len(load(path,expected)[0])==2
    with pytest.raises(ValueError,match="Incompatible"):
        load(path,{**expected,"rounds":2})
    path.write_text("tampered",encoding="utf-8")
    with pytest.raises(ValueError,match="digest"):
        load(path,expected)
    with pytest.raises(ValueError,match="order"):
        validate_order(rows.iloc[::-1],expected)
    rows.iloc[:1].to_csv(path,index=False)
    publish(path,expected,complete=False)
    assert len(load(path,expected,complete=False)[0])==1
    with pytest.raises(ValueError,match="completion"):
        load(path,expected)


def test_equivalent_raw_threshold_selects_same_detection_ids():
    records = {21: {"detections": [(3, 1, .28), (4, 1, .33)]},
               19: {"detections": [(3, 1, .5), (7, 1, .29)]}}
    threshold, original, scaled, equivalent = post_nms_cohorts(records, .25, .75)
    assert threshold == pytest.approx(1 / (1 + 3 ** .75))
    assert original == [(19, 3), (19, 7), (21, 3), (21, 4)]
    assert scaled == equivalent == [(19, 3), (21, 4)]


def test_operating_point_and_nested_partitions():
    out=operating_row(np.array([.25,.9,.1]),np.array([0,1,1]),.25,4,3)
    assert (out["tp"],out["fp"],out["fn"],out["ignored"],out["recall"])==(1,1,3,1,.25)
    c,e=list(range(200)),list(range(200,5000))
    validate_partitions(dict(calib_200=c,eval_4800=e,eval_1000=e[:1000]))
    with pytest.raises(ValueError,match="partitions"):
        validate_partitions(dict(calib_200=c,eval_4800=e,eval_1000=[1]+e[:999]))
