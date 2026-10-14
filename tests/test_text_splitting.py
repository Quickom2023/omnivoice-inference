"""split_text_for_streaming — where long text is cut before it reaches the model.

Every chunk is generated as a finished utterance: the model lets the voice
fall at its end and the post-processing fades it out. A cut is therefore
only harmless where a reader would pause anyway, so the splitter prefers
sentence ends, then clause punctuation, and uses a bare word boundary only
when there is no punctuation to cut at.
"""

from __future__ import annotations

import pytest

from streaming_api_omnivoice import OmniVoiceStreamingService

# An instance, because the splitter is an instance method. Constructing the
# service builds a scheduler but never starts it, so no model is loaded.
SPLIT = OmniVoiceStreamingService().split_text_for_streaming

# 90 characters; its first two clauses together are 55.
LONG_SENTENCE = (
    "Hôm nay trời rất đẹp{sep} chúng tôi đi dạo trong công viên{sep} "
    "sau đó ghé quán cà phê quen thuộc."
)


@pytest.mark.parametrize("sep", [",", ";", ":"])
def test_a_long_sentence_is_cut_at_its_last_clause_break_that_fits(sep):
    # A word-boundary cut would end the first chunk on "sau", mid-phrase.
    assert SPLIT(LONG_SENTENCE.format(sep=sep), max_chars=60) == [
        f"Hôm nay trời rất đẹp{sep} chúng tôi đi dạo trong công viên{sep}",
        "sau đó ghé quán cà phê quen thuộc.",
    ]


def test_the_tail_of_a_cut_sentence_is_packed_with_the_next_sentence():
    # Leaving the tail on its own would add a chunk, and a seam, for nothing.
    text = LONG_SENTENCE.format(sep=",") + " Mọi người đều vui."
    assert SPLIT(text, max_chars=60) == [
        "Hôm nay trời rất đẹp, chúng tôi đi dạo trong công viên,",
        "sau đó ghé quán cà phê quen thuộc. Mọi người đều vui.",
    ]


def test_commas_in_a_sentence_that_fits_are_not_cut():
    # Commas are a fallback for over-long sentences, not a cut point of their own.
    text = "Trời đẹp, gió nhẹ. Chúng tôi đi dạo."
    assert SPLIT(text, max_chars=240) == [text]


def test_a_decimal_comma_is_not_a_clause_break():
    # Cutting at "12," would read the number as "twelve, five percent".
    text = "Lãi suất tăng 12,5% so với cùng kỳ năm ngoái, mức cao nhất trong nhiều năm qua."
    assert SPLIT(text, max_chars=50) == [
        "Lãi suất tăng 12,5% so với cùng kỳ năm ngoái,",
        "mức cao nhất trong nhiều năm qua.",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Doanh thu đạt 1.250.000 tỷ đồng.",
        "Giá vàng tăng 3.5 lần.",
        "Tôi sống ở TP.HCM từ nhỏ.",
        "Xem chi tiết tại example.com nhé.",
    ],
)
def test_a_dot_not_followed_by_a_space_does_not_end_a_sentence(text):
    # Treating it as a sentence end re-joins the pieces with a space,
    # so "1.250.000" would be spoken as "1. 250. 000".
    assert SPLIT(text, max_chars=240) == [text]


def test_a_sentence_still_ends_before_a_newline():
    assert SPLIT("Câu thứ nhất.\nCâu thứ hai.", max_chars=20) == [
        "Câu thứ nhất.",
        "Câu thứ hai.",
    ]


def test_text_without_punctuation_falls_back_to_word_boundaries():
    text = "một hai ba bốn năm sáu bảy tám chín mười"
    assert SPLIT(text, max_chars=20) == [
        "một hai ba bốn năm",
        "sáu bảy tám chín",
        "mười",
    ]


PARAGRAPH = (
    "Trong lịch sử, nhiều ngành khoa học như vật lý học và tâm lý học từng là bộ phận "
    "của triết học, nhưng ngày nay được xem như là những môn học thuật riêng biệt theo "
    "cách hiểu hiện đại của thuật ngữ, với phương pháp và đối tượng nghiên cứu riêng, "
    "được giảng dạy ở các khoa độc lập trong trường đại học. Theo báo cáo năm 2024, "
    "doanh thu đạt 1.250.000 tỷ đồng, tăng 12,5% so với cùng kỳ. Một số nền văn minh "
    "triết học có tầm ảnh hưởng gồm triết học phương Tây, Hồi giáo, Ấn Độ và Trung Quốc."
)


@pytest.mark.parametrize("max_chars", [60, 120, 240])
def test_chunks_fit_the_limit_and_keep_every_word_in_order(max_chars):
    chunks = SPLIT(PARAGRAPH, max_chars=max_chars)
    assert all(len(chunk) <= max_chars for chunk in chunks), chunks
    assert " ".join(chunks).split() == PARAGRAPH.split()
