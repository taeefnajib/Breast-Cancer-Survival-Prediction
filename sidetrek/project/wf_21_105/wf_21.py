import catboost
import os
import typing
from flytekit import workflow
from project.wf_21_105.wf import Hyperparameters
from project.wf_21_105.wf import run_wf

_wf_outputs=typing.NamedTuple("WfOutputs",run_wf_0=catboost.core.CatBoostClassifier)
@workflow
def wf_21(_wf_args:Hyperparameters)->_wf_outputs:
	run_wf_o0_=run_wf(hp=_wf_args)
	return _wf_outputs(run_wf_o0_)