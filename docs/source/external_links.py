"""
    sphinx.ext.extlinks
    ~~~~~~~~~~~~~~~~~~~
    Extension to save typing and prevent hard-coding of base URLs in the reST
    files.
    This adds a new config value called ``extlinks`` that is created like this::
       extlinks = {'exmpl': ('https://example.invalid/%s.html', caption), ...}
    Now you can use e.g. :exmpl:`foo` in your documents.  This will create a
    link to ``https://example.invalid/foo.html``.  The link caption depends on
    the *caption* value given:
    - If it is ``None``, the caption will be the full URL.
    - If it is a string, it must contain ``%s`` exactly once.  In this case the
      caption will be *caption* with the role content substituted for ``%s``.
    You can also give an explicit caption, e.g. :exmpl:`Foo <foo>`.
    Both, the url string and the caption string must escape ``%`` as ``%%``.
    :copyright: Copyright 2007-2021 by the Sphinx team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

from typing import Any, Dict, List, Tuple

import sphinx
from docutils import nodes, utils
from docutils.nodes import Node, system_message
from docutils.parsers.rst.states import Inliner
from sphinx.application import Sphinx
from sphinx.util import caption_ref_re

MODEL_TYPE_MAPPING = {
    "acoustic": "acoustic model",
    "g2p": "g2p model",
    "lm": "language model",
    "dictionary": "dictionary",
    "ivector": "ivector extractor",
}


def model_role(
    typ: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: Inliner,
    options: Dict = None,
    content: List[str] = None,
) -> Tuple[List[Node], List[system_message]]:
    text = utils.unescape(text)
    model_type, model_name = text.split("/")
    full_url = f"https://github.com/MontrealCorpusTools/mfa-models/raw/main/{model_type}/{model_name.lower()}.zip"
    title = f"{model_name.title()} {MODEL_TYPE_MAPPING[model_type]}"
    pnode = nodes.reference(title, title, internal=False, refuri=full_url)
    return [pnode], []


def kaldi_steps_role(
    typ: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: Inliner,
    options: Dict = None,
    content: List[str] = None,
) -> Tuple[List[Node], List[system_message]]:
    text = utils.unescape(text)
    full_url = f"https://github.com/kaldi-asr/kaldi/tree/master/egs/wsj/s5/steps/{text}.sh"
    title = f"{text}.sh"
    pnode = nodes.reference(title, title, internal=False, refuri=full_url)
    return [pnode], []


def kaldi_steps_sid_role(
    typ: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: Inliner,
    options: Dict = None,
    content: List[str] = None,
) -> Tuple[List[Node], List[system_message]]:
    text = utils.unescape(text)
    full_url = f"https://github.com/kaldi-asr/kaldi/tree/cbed4ff688a172a7f765493d24771c1bd57dcd20/egs/sre08/v1/sid/{text}.sh"
    title = f"sid/{text}.sh"
    pnode = nodes.reference(title, title, internal=False, refuri=full_url)
    return [pnode], []


def kaldi_src_role(
    typ: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: Inliner,
    options: Dict = None,
    content: List[str] = None,
) -> Tuple[List[Node], List[system_message]]:
    text = utils.unescape(text)
    full_url = f"https://github.com/kaldi-asr/kaldi/tree/master/src/{text}.cc"
    title = f"{text}.cc"
    pnode = nodes.reference(title, title, internal=False, refuri=full_url)
    return [pnode], []


def xref(
    typ: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: Inliner,
    options: Dict = None,
    content: List[str] = None,
) -> Tuple[List[Node], List[system_message]]:

    title = target = text
    # look if explicit title and target are given with `foo <bar>` syntax
    brace = text.find("<")
    if brace != -1:
        m = caption_ref_re.match(text)
        if m:
            target = m.group(2)
            title = m.group(1)
        else:
            # fallback: everything after '<' is the target
            target = text[brace + 1 :]
            title = text[:brace]

    link = xref.links[target]

    if brace != -1:
        pnode = nodes.reference(target, title, refuri=link[1])
    else:
        pnode = nodes.reference(target, link[0], refuri=link[1])

    return [pnode], []


def get_refs(app):
    xref.links = app.config.xref_links


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_config_value("xref_links", {}, "env")
    app.add_role("mfa_model", model_role)
    app.add_role("kaldi_steps", kaldi_steps_role)
    app.add_role("kaldi_steps_sid", kaldi_steps_sid_role)
    app.add_role("kaldi_src", kaldi_src_role)
    app.add_role("xref", xref)
    app.connect("builder-inited", get_refs)
    return {"version": sphinx.__display_version__, "parallel_read_safe": True}
