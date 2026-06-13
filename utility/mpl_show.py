# -*- coding: utf-8 -*-
"""Matplotlib display helpers: show in IPython, skip blocking windows for plain scripts."""


def in_ipython_session() -> bool:
    """True when running inside IPython or Jupyter (not plain ``python script.py``)."""
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def resolve_show(show=None) -> bool:
    """Resolve whether to call plt.show().

    Args:
        show: None = auto (IPython only), True/False = explicit override.
    """
    if show is not None:
        return bool(show)
    return in_ipython_session()


def show_or_close(show=None) -> None:
    """Show the current figure in IPython; close it in plain terminal runs."""
    import matplotlib.pyplot as plt

    if resolve_show(show):
        plt.show()
    else:
        plt.close()
