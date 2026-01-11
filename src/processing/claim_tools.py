import re

class ClaimExtractor:
    """
    Extracts atomic claims from a backstory using sentence splitting heuristics.
    Also extracts entities like Dates for counter-factual retrieval.
    """
    def __init__(self):
        # Basic sentence splitters: ., ?, ! followed by space or end of string
        self.split_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        # Date pattern: 1700-1999
        self.date_pattern = r'\b(17|18|19)\d{2}\b'

    def extract_claims(self, text):
        """
        Splits text into a list of atomic claims (sentences).
        """
        if not text:
            return []

        # Clean text
        text = text.strip().replace('\n', ' ')

        # Split
        claims = re.split(self.split_pattern, text)

        # Filter empty or too short
        claims = [c.strip() for c in claims if len(c.strip()) > 10]

        if not claims:
            return [text] # Fallback to whole text

        return claims

    def extract_dates(self, text):
        """
        Extracts year-like tokens from text.
        """
        if not text:
            return []
        return re.findall(self.date_pattern, text)

if __name__ == "__main__":
    extractor = ClaimExtractor()
    text = "Thalcave’s people faded as colonists advanced. His father knew the pampas geography. In 1852, boyhood was spent roaming the plains."
    print("Claims:", extractor.extract_claims(text))
    print("Dates:", extractor.extract_dates(text))
