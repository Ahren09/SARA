import re

import regex


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def extract_entities_and_numerics(nlp, text):
    """Extract NER entities and numerics from text using a spaCy nlp pipeline."""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'GPE', 'EVENT', 'CATEGORY']]
    numerics = [ent.text for ent in doc.ents if ent.label_ == 'CARDINAL' or re.match(r'\b\d+\b', ent.text)]
    return set(entities), set(numerics)


