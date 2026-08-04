"""One shared, offline, deterministic answer classifier.

Three copies of "did they say yes or no?" used to live in the codebase
(agent/corroboration.py, VoiceAgent._affirmation, and an inline split inside
VoiceAgent._consume_heard), each with a different, narrower vocabulary. This
module is the single implementation they all delegate to, with the union of
their vocabularies plus the natural phrasings people actually use when
declining or accepting a gentle offer.

Everything here is pure, local, and synchronous: answer interpretation runs
inside the frame loop, so it must never touch the network.

`PendingClassification` below is the one exception's bookkeeping, not an
exception itself: when the keywords come back "unclear" the agent may ask the
configured model to refine that single verdict on a *later* tick. The keywords
still decide — see VoiceAgent._poll_classification.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

VERDICTS = ("affirmed", "denied", "unclear")

# How long a speculative refinement may stay useful. Deliberately far below the
# provider's 8 s HTTP timeout: a verdict that lands later than this is answering
# a question the conversation has already moved past.
CLASSIFICATION_DEADLINE = 3.0


@dataclass
class PendingClassification:
    """One in-flight answer classification, bound to the exact question asked."""
    request_id: str
    lane: str            # "action" | "corroboration" | "workflow" | "skin"
    question_id: str
    target: str
    question: str
    text: str            # session-only; never persisted, never logged, never in diagnostics
    submitted_at: float
    deadline: float      # submitted_at + 3.0
    heard_at: float

# Words and phrases that (dis)confirm a health follow-up or an offered action
# in casual speech. Denials are checked FIRST: "no, not really" must not
# confirm via "really", and "no thanks" must not confirm via a stray token.
_DENY = (
    # --- historic corroboration vocabulary (verbatim)
    "no", "nope", "not really", "nothing", "haven't", "hasn't", "don't",
    "doesn't", "i'm fine", "im fine", "i am fine", "all good", "never",
    # --- merged in from VoiceAgent._affirmation
    "cancel", "stop", "dont",
    # --- merged in from the inline workflow-answer split (keeps that lane's
    # vocabulary from shrinking)
    "none",
    # --- natural ways of declining an offer without saying "no"
    "not now", "maybe later", "rather not", "i'd rather not", "id rather not",
    "no thanks", "no thank you", "not today", "leave it", "another time",
)
_CONFIRM = (
    # --- historic corroboration vocabulary (verbatim)
    "yes", "yeah", "yep", "a bit", "a little", "i have", "i do",
    "actually", "lately", "sometimes", "now that you mention",
    "i guess so", "kind of", "sort of",
    # --- merged in from VoiceAgent._affirmation
    "okay", "ok", "sure",
    # --- natural ways of accepting an offer without saying "yes"
    "go ahead", "why not", "please do", "sounds good", "alright",
    "of course", "definitely", "that would be nice", "i would like",
)

# See interpret_answer: bare "please" only affirms in a reply this short.
_PLEASE_MAX_WORDS = 3


def interpret_answer(text: str) -> str:
    """Classify a spoken reply as one of VERDICTS, offline and deterministically.

    Punctuation is stripped and the text padded with spaces so every match is
    whole-word bounded ("no" must not fire inside "noticed"). Denials are
    checked before confirmations so "no, not really" denies rather than
    confirming on a stray token.
    """
    t = " " + re.sub(r"[^a-z' ]+", " ", text.lower()) + " "
    for w in _DENY:
        if f" {w} " in t:
            return "denied"
    for w in _CONFIRM:
        if f" {w} " in t:
            return "affirmed"
    # Bare "please" is an affirmation only in a very short reply ("please",
    # "yes please", "please do that"). In anything longer it is an imperative
    # naming a DIFFERENT action -- "please check my arm instead" -- and the
    # caller checks a pending proposal before it maps speech to a new one, so
    # affirming here would start the wrong action. Fall through to "unclear".
    if " please " in t and len(t.split()) <= _PLEASE_MAX_WORDS:
        return "affirmed"
    return "unclear"
