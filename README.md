# Processing and normalizing text from the Gutenberg corpus
Processes the [Gutenberg corpus](https://drive.google.com/file/d/17WBziFbt9nhAW5iV-yHPHmCfquBPrjJO/view) in parallel.

Output is segmented, and tokenized by default, i.e. one tokenized sentence per line.
Optionally, a max and min value can be specified for the length of the sentences.

## Usage

```
usage: segment-books.py [-h] [--max-tokens MAX_TOKENS]
                        [--min-tokens MIN_TOKENS] [-n N_JOBS]
                        [--no-tokenized-output] [--remove-emphasis]
                        [--remove-notes] [--spacy-model SPACY_MODEL]
                        din dout

Process the Gutenberg corpus. Most importantly, segment the books so that
there is one sentence per line.

positional arguments:
  din                   input directory. All files in all subdirectories will
                        be processed.
  dout                  output directory.

optional arguments:
  -h, --help            show this help message and exit
  --max-tokens MAX_TOKENS
                        sentences with more than 'max_tokens' won't be
                        included in the output. (default: None)
  --min-tokens MIN_TOKENS
                        sentences with less than 'min_tokens' won't be
                        included in the output. (default: None)
  -n N_JOBS, --n-jobs N_JOBS
                        number of workers to use (default depends on your
                        current system; core count - 1). (default: 3)
  --no-tokenized-output
                        do not tokenize the articles. (default: False)
  --remove-emphasis     remove emphasis characters such as the underscore in
                        '_emphasis_'. (default: False)
  --remove-notes        remove notes such as '[Illustration]'. (default:
                        False)
  --spacy-model SPACY_MODEL
                        spaCy model to use for sentence segmentation.
                        (default: en_core_web_sm)
```

## Requirements (cf. Pipfile)
 - Python 3.6 or higher
 - [spaCy](https://spacy.io/usage/)
 - a spaCy [language model](https://spacy.io/usage/models) (en_core_web_sm by default)

## Important note
Because of how I implemented the sentence boundary detection through spaCy, consecutive direct speech clauses are not segmented. 
What this means is that a sentence such as 

> Wilson said of the "Times"' owner, the Journal Register Company, "They don't care about the product. They don't care about the customer. They don't care about the employees. And they don't know anything about the business."

will be parsed as one single sentence.

For more information, see [this issue #3553](https://github.com/explosion/spaCy/issues/3553).
