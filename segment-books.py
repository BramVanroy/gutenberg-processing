from concurrent.futures import ProcessPoolExecutor
import logging
from math import inf
from os import cpu_count
from pathlib import Path
import re
import time

import spacy

logging.basicConfig(datefmt='%d-%b %H:%M:%S',
                    format='%(asctime)s - [%(levelname)s]: %(message)s',
                    level=logging.INFO)

"""
    Processes the Gutenberg corpus in parallel.

    Output is segmented, and tokenized by default, i.e. one tokenized sentence per line.
    Optionally, a max and min value can be specified for the length of the sentences.
"""

DEFAULT_WORKERS = (cpu_count() - 1) or 1


class BookSegmentor:
    def __init__(self,
                 max_tokens=None,
                 min_tokens=None,
                 n_jobs=DEFAULT_WORKERS,
                 no_tokenized_output=False,
                 remove_emphasis=False,
                 remove_notes=False,
                 spacy_model='en_core_web_sm'):
        self.max_tokens = max_tokens if max_tokens else inf
        self.min_tokens = min_tokens if min_tokens else 0
        self.n_jobs = n_jobs
        self.no_tokenized_output = no_tokenized_output
        self.remove_emphasis = remove_emphasis
        self.emphasis_regex = re.compile(r'[*_|]+(.*?)[*_|]+') if remove_emphasis else None
        self.remove_notes = remove_notes

        self.nlp = spacy.load(spacy_model, disable=['ner', 'textcat'])
        self.nlp.add_pipe(BookSegmentor.prevent_wrapped_sbd, name='prevent-wrapped-sbd', before='parser')
        logging.info(f"Using spaCy model '{spacy_model}'")

        self.pdin = None
        self.pdout = None

    def extract_articles(self, din, dout):
        """
        Iterate over all subdirectories and process all files with 'process_file'
        """
        self.pdin = Path(din).resolve()
        self.pdout = Path(dout).resolve()

        start_time = time.time()

        total_books_n = 0
        total_lines_n = 0
        files = (pfin for pfin in self.pdin.rglob('*.txt') if pfin.is_file())

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            logging.info(f"Processing dir {str(self.pdin)} with {self.n_jobs} workers...")
            for filename, lines_n in executor.map(self.process_file, files):
                total_books_n += 1
                total_lines_n += lines_n
                logging.info(f"Wrote {lines_n} lines from file {filename}...")

        logging.info(f"Finished! Wrote {total_books_n} books ({total_lines_n} lines)"
                     f" in {time.time() - start_time:.0F} seconds.")

    def process_file(self, pfin):
        """
        Process all lines in a file with 'parse_json'.
        One JSON object per line.
        """
        lines_n = 0
        with open(pfin, 'r', encoding='utf-8') as fhin:
            text = fhin.read()
            lines_n += self.process_text(text, pfin.name)

        return pfin.name, lines_n

    def process_text(self, text, filename):
        """ Given raw text, process it as required: remove headings and/or segment it with 'segment_text'."""

        lines = self.segment_text(text)

        lines_n = len(lines) if lines else 0
        text = '\n'.join(lines) if lines else None

        # 'text' can be None, e.g. due to a max-tokens value
        if text:
            filename = self.pdout.joinpath(filename)
            with open(filename, 'w', encoding='utf-8') as fhout:
                fhout.write(text)

        return lines_n

    def segment_text(self, text):
        """ Segment text into sentences. If required, also tokenize the output."""
        text = text.replace('. . .', '... ')

        if self.remove_emphasis:
            text = re.sub(self.emphasis_regex, '\\1', text)

        # Split on new line and remove empty lines
        paragraphs = text.split('\n\n')
        # Get rid of multiple white-space and put lines in list
        lines = list(filter(None, [' '.join(p.split()) for p in paragraphs]))
        # Process with spaCy
        docs = list(self.nlp.pipe(lines))
        spacy_sents = [sent for doc in docs for sent in doc.sents]

        # Filter too long or too short sentences
        spacy_sents = [sent for sent in spacy_sents if self.min_tokens <= len(sent) <= self.max_tokens]

        # Export as tokenized output
        if not self.no_tokenized_output:
            # spacy_sents in fact already contains the Tokens objects.
            # We just need to split and join with white space
            sents = []
            for sent in spacy_sents:
                sentence_tokenized = ' '.join([token.text for token in sent])
                # Get rid of multiple white-space
                sentence_tokenized = ' '.join(sentence_tokenized.split())
                sents.append(sentence_tokenized)
        else:
            # Just keep the sentence representations as-is, without separated tokens
            sents = [sent.text for sent in spacy_sents]

        if self.remove_notes:
            sents = [s for s in sents if not (s.strip().startswith('[') and s.strip().endswith(']'))]

        return sents

    @staticmethod
    def prevent_wrapped_sbd(doc):
        """ spaCy's SBD sees ending quotation marks as a separate sentence.
            Ensure that SBD does not run on tokens inside quotation marks and brackets.
            See this issue: https://github.com/explosion/spaCy/issues/3553
        """
        quote_open = False
        bracket_open = False
        can_sbd = True
        for token in doc:
            # Don't do sbd on these tokens
            if not can_sbd:
                token.is_sent_start = False

            # Not using .is_quote so that we don't mix-and-match different kinds of quotes (e.g. ' and ")
            # Especially useful since quotes don't seem to work well with .is_left_punct or .is_right_punct
            if token.text == '"':
                quote_open = False if quote_open else True
            elif token.is_bracket and token.is_left_punct:
                bracket_open = True
            elif token.is_bracket and token.is_right_punct:
                bracket_open = False

            can_sbd = not (quote_open or bracket_open)

        return doc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process the Gutenberg corpus. Most importantly, segment the books'
                                                 ' so that there is one sentence per line.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('din', help='input directory. All .txt files in all subdirectories will be processed.')
    parser.add_argument('dout', help='output directory.')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help="sentences with more than 'max_tokens' won't be included in the output.")
    parser.add_argument('--min-tokens', type=int, default=None,
                        help="sentences with less than 'min_tokens' won't be included in the output.")
    parser.add_argument('-n', '--n-jobs', type=int, default=DEFAULT_WORKERS,
                        help=f"number of workers to use (default depends on your current system; core count - 1).")
    parser.add_argument('--no-tokenized-output', action='store_true', default=False,
                        help="do not tokenize the articles.")
    parser.add_argument('--remove-emphasis', action='store_true', default=False,
                        help="remove emphasis characters such as the underscore in '_emphasis_'.")
    parser.add_argument('--remove-notes', action='store_true', default=False,
                        help="remove notes such as '[Illustration]'.")
    parser.add_argument('--spacy-model', default='en_core_web_sm',
                        help='spaCy model to use for sentence segmentation.')

    args = parser.parse_args()

    extractor = BookSegmentor(args.max_tokens,
                              args.min_tokens,
                              args.n_jobs,
                              args.no_tokenized_output,
                              args.remove_emphasis,
                              args.remove_notes,
                              args.spacy_model)
    extractor.extract_articles(args.din, args.dout)
