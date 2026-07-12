import pytest
from rag.answer import (construct_answer, detect_corpus_boundary,
                        split_sentences, REFUSAL_INSUFFICIENT, BOUNDARY_MCM,
                        BOUNDARY_OUT_OF_CORPUS)


def chunk(number, title, text, ctype="punitive", part=0):
    return {"id": f"sec_{number}.txt::{part}", "number": number,
            "title": title, "text": text, "type": ctype,
            "citation": f"Art. {number}, UCMJ"}


ART_25A = chunk("25a", "Number of members in capital cases",
                "(a) In General.— In a case in which the accused may be "
                "sentenced to death, the number of members shall be 12.",
                ctype="procedural")
ART_86 = chunk("86", "Absence without leave",
               "Any member of the armed forces who, without authority— "
               "(1) fails to go to his appointed place of duty at the time "
               "prescribed; (2) goes from that place; or (3) absents himself "
               "or remains absent from his unit, organization, or place of "
               "duty at which he is required to be at the time prescribed; "
               "shall be punished as a court-martial may direct.")
OFF_TOPIC = chunk("77", "Principals",
                  "Any person punishable under this chapter who commits an "
                  "offense punishable by this chapter, or aids, abets, "
                  "counsels, commands, or procures its commission.")


def test_deterministic():
    retrieved = [(ART_86, 9.0), (OFF_TOPIC, 3.0)]
    q = "When is a member absent without leave from his place of duty?"
    a = construct_answer(q, retrieved)
    b = construct_answer(q, retrieved)
    assert a == b


def test_refuses_on_insufficient_evidence():
    out = construct_answer("What is the statute of limitations for perjury?",
                           [(OFF_TOPIC, 1.0)])
    assert out["refused"] is True
    assert out["text"] == REFUSAL_INSUFFICIENT
    assert out["cited_articles"] == []


def test_exact_number_preserved():
    out = construct_answer(
        "How many members sit on a court-martial panel in a capital case?",
        [(ART_25A, 8.0), (OFF_TOPIC, 1.0)], qtype="numeric")
    assert not out["refused"]
    assert "shall be 12" in out["text"]
    assert out["cited_articles"] == ["25a"]


def test_citations_attached():
    out = construct_answer(
        "When is a member absent without leave from his place of duty?",
        [(ART_86, 9.0)])
    assert "(Art. 86, UCMJ)" in out["text"]


def test_mcm_boundary_on_punishment_amount():
    # the offense article says 'as a court-martial may direct'; a maximum
    # confinement question must hit the MCM boundary, not invent a figure
    out = construct_answer(
        "What is the maximum confinement for absence without leave?",
        [(ART_86, 9.0)])
    assert out["boundary"] is True
    assert BOUNDARY_MCM in out["text"]


def test_punishment_amount_never_answered_from_procedural_day_limits():
    art15 = chunk("15", "Commanding officer's non-judicial punishment",
                  "(1) correctional custody for not more than 30 consecutive "
                  "days; (2) forfeiture of not more than seven days' pay.",
                  ctype="procedural")
    out = construct_answer(
        "What is the maximum confinement for being AWOL for more than 30 days?",
        [(art15, 9.0)])
    assert out["refused"] is True
    assert out["boundary"] is True
    assert "30" not in out["text"].replace("chapter 47", "")


def test_out_of_corpus_boundary():
    assert detect_corpus_boundary(
        "What does Rule for Courts-Martial 707 require for a speedy trial?")
    assert not detect_corpus_boundary("Who is subject to the UCMJ?")
    out = construct_answer(
        "What does Rule for Courts-Martial 707 require for a speedy trial?",
        [(ART_86, 5.0)])
    assert out["refused"] and out["boundary"]
    assert out["text"] == BOUNDARY_OUT_OF_CORPUS


def test_permissive_always_answers():
    out = construct_answer("What is the statute of limitations for perjury?",
                           [(OFF_TOPIC, 1.0)], policy="permissive")
    assert out["refused"] is False
    assert out["cited_articles"] == ["77"]


def test_sentences_are_verbatim_extracts():
    out = construct_answer(
        "When is a member absent without leave from his place of duty?",
        [(ART_86, 9.0)])
    flowed = " ".join(ART_86["text"].split())
    for s in out["sentences"]:
        assert " ".join(s["text"].split()) in flowed


def test_split_drops_history_notes():
    text = ("Any member who deserts shall be punished as a court-martial "
            "may direct. (Aug. 10, 1956, ch. 1041, 70A Stat. 67 ; Pub. L. "
            "96–513, title V, § 511(25) , Dec. 12, 1980 , 94 Stat. 2922 .)")
    sents = split_sentences(text)
    assert len(sents) == 1
    assert "Stat." not in sents[0]
