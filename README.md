# Natural Language Processing HW2

Please see https://www3.nd.edu/~dchiang/teaching/nlp/hw2.html for instructions.

## Data

The data has four parts:
- `small.zh-en`: small training data, Star Wars Episodes 4-6 and 1-2
- `large.zh-en`: large training data, various movies
- `dev.zh`, `dev.reference.en`, `dev.backstroke.en`: development data, first part of Star Wars Episode 3
- `test.zh`, `test.reference.en`, `test.backstroke.en`: test data, remainder of Star Wars Episode 3

The suffixes mean:
- `*.zh`: Chinese, one sentence per line
- `*.reference.en`: original English subtitles, one sentence per line
- `*.backstroke.en`: _Backstroke of the West_ English subtitles, one sentence per line
- `*.zh-en`: Chinese-English, one tab-separated sentence pair per line

## Sources

The small training data and the dev/test data are from opensubtitles.org.

Preprocessing:
- Chinese:
  - Convert from GBK to UTF8
  - Split Chinese characters, but leave Latin characters
- English text
  - Convert from CP1252 to UTF8
  - Split punctuation marks, but keep contracted words like n't, 'll, etc., together.
- Sentence alignment was performed using timestamps.

The large training data is from the OpenSubtitles portion of OPUS
(which is also derived from opensubtitles.org). We selected all films
released between 2000 and 2004.
