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

        # Keywords for Type Classification
        self.hard_fact_keywords = [
            "born", "died", "death", "birth", "father", "mother", "sister", "brother", "son", "daughter",
            "married", "wedding", "wife", "husband", "arrested", "prison", "jail", "killed", "murdered"
        ]

        self.common_starts = {"The", "A", "An", "In", "On", "At", "He", "She", "It", "They", "We", "You", "But", "And", "However", "Although", "Later", "Then"}

    def extract_entities(self, text):
        """
        Extracts potential Named Entities (Capitalized words not at start of sentence, or known locations).
        Heuristic only since no spaCy.
        """
        if not text: return []

        # Regex for capitalized phrases
        # We try to catch "Edmond Dantes" or "Paris"
        # We skip words that are likely just sentence starters if they appear at the very beginning

        # 1. Find all capitalized words/phrases
        pattern = r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'
        matches = set(re.findall(pattern, text))

        # 2. Filter out common stopwords if they are single words
        cleaned = []
        for m in matches:
            if ' ' not in m and m in self.common_starts:
                continue
            cleaned.append(m)

        # 3. Add Hardcoded Locations if present (case insensitive check, return titled)
        for loc in self.locations:
            if loc.lower() in text.lower():
                cleaned.append(loc)

        return list(set(cleaned))

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

    def classify_claim(self, claim):
        """
        Classifies claim as 'Type A' (Hard Fact) or 'Type B' (Soft State).
        """
        text_lower = claim.lower()

        # Check for Dates
        if self.extract_dates(claim):
            return 'Type A'

        # Check for Hard Fact Keywords
        if any(kw in text_lower for kw in self.hard_fact_keywords):
            return 'Type A'

        # Check for Specific Locations (implies Spatiotemporal assertion)
        if any(loc.lower() in text_lower for loc in self.locations):
            return 'Type A'

        return 'Type B'

if __name__ == "__main__":
    extractor = ClaimExtractor()
    text = "In 1815, Thalcave was in Paris. Later he felt sad about his father."
    print("Claims:", extractor.extract_claims(text))
    for c in extractor.extract_claims(text):
        print(f"Claim: '{c}' -> {extractor.classify_claim(c)}")
