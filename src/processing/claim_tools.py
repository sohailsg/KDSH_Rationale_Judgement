import re

class ClaimExtractor:
    """
    Extracts atomic claims from a backstory using sentence splitting heuristics.
    Ideally replaces with an LLM-based extractor if available.
    """
    def __init__(self):
        # Basic sentence splitters: ., ?, ! followed by space or end of string
        self.split_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'

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

if __name__ == "__main__":
    extractor = ClaimExtractor()
    text = "Thalcave’s people faded as colonists advanced. His father knew the pampas geography. Boyhood was spent roaming the plains."
    print(extractor.extract_claims(text))
