examples = [
    {"q": "Which example describes a learned behavior in a dog?",
     "a": ["smelling the air for odors", "barking when disturbed", "sitting on command", "digging in soil"],
     "correct": 2},
    {"q": "Which substance should a student apply to the skin if he or she gets splashed with an acid?",
     "a": ["water", "vinegar", "salt", "formaldehyde"],
     "correct": 0},
    {"q": "Over time, non-volcanic mountains can form due to the interaction of plate boundaries. Which interaction is most likely associated with the formation of non-volcanic mountains?",
     "a": ["oceanic plates colliding with oceanic plates", "oceanic plates separating from oceanic plates", "continental plates colliding with continental plates", "continental plates separating from continental plates"],
     "correct": 2},
]

def extractFocus(question, answers):
    None

def extractSemanticVector(answer):
    None

model = gensim.models.Word2Vec.load_word2vec_format('../../../Word2Vec/GoogleNews-vectors-paraphrase-300.bin', binary=True)
model.init_sims(replace=True)
print("Loaded word2vec model")