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
        # Location keywords (Hardcoded for 19th century context, expandable)
        self.locations = ["Paris", "London", "Rome", "Marseille", "Château d’If", "prison", "dungeon", "sea", "India", "Spain", "Italy"]

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

    def extract_anchors(self, text):
        """
        Extracts Spatiotemporal Anchors: (Date, Location).
        Returns list of dicts.
        """
        anchors = []
        dates = self.extract_dates(text)

        # Simple extraction: if sentence has date and location
        claims = self.extract_claims(text)
        for claim in claims:
            c_dates = self.extract_dates(claim)
            c_locs = [loc for loc in self.locations if loc.lower() in claim.lower()]

            if c_dates:
                for d in c_dates:
                    anchors.append({'date': d, 'locations': c_locs, 'claim': claim})

        return anchors

if __name__ == "__main__":
    extractor = ClaimExtractor()
    text = "In 1815, Thalcave was in Paris. Later he went to Rome."
    print("Claims:", extractor.extract_claims(text))
    print("Anchors:", extractor.extract_anchors(text))
