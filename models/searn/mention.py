from typing import List


# Class to describe mention
class Mention:

    # Tokens of the entity
    tokens = []

    # ID of cluster
    cluster_id: str = ""

    # Check if it is an entity
    is_entity: bool = False

    def __init__(self, _tokens):
        self.tokens = _tokens.copy()


class Word:
    ID = None
    RawText = None
    DocumentID = None
    WordOrder = None
    PartOfSpeech = None
    Lemmatized = None
    IsPlural = None
    IsProperName = None
    IsHeadWord = None
    Gender = None
    EntityID = None
    RawTagString = None
    CoreferenceGroupID = None
    
