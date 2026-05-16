"""Training and evaluation entry points.

These modules import torch at the top of their own files; we do *not*
re-export them eagerly here. Otherwise ``import lhfm.training`` would crash
inside a torch-less environment (Sphinx build, CI lint job, ``--help``
introspection on a fresh CPU container).

Consumers should write::

    from lhfm.training.train_ssl import pretrain_ssl
    from lhfm.training.train_downstream import train_downstream

which only pulls torch when the relevant code path is actually exercised.
"""

__all__: list[str] = []
