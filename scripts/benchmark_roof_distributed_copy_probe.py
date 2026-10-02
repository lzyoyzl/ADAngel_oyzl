#!/usr/bin/env python3
"""v64: unchanged two-partial G128 math, four-phase asynchronous copy screen."""
from benchmark_roof_interleaved_merge_probe import (
    Driver as BaseDriver, checked_cubins as check, main as run, summary as summarize,
)
from probe_roof_distributed_copy_codegen import generated_header


def checked_cubins(directory):
    return check(directory,transform=generated_header,stem_prefix='distributed_copy')


class Driver(BaseDriver):
    def __init__(self,*args):
        super().__init__(*args)
        for policy,info in self.resources.items():
            info.pop('interleaved_merge')
            info['partial_registers_per_four_n_atoms']=32
            info['distributed_copy']=policy
            info['copy_issue_phases']=4 if policy else 1


def summary(rows):
    return summarize(rows,'distributed_copy')


if __name__=='__main__':
    run(cubin_checker=checked_cubins,driver_factory=Driver,
        policy_key='distributed_copy',description=__doc__)
