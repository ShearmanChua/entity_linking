import spacy

nlp = spacy.load("en_core_web_trf")
doc = nlp("Ada Lovelace was born in London")
print(nlp.vocab[0])

# Document level
ents = [(e.text, e.label_, e.kb_id_) for e in doc.ents]
print(ents)  # [('Ada Lovelace', 'PERSON', 'Q7259'), ('London', 'GPE', 'Q84')]

# Token level
ent_ada_0 = [doc[0].text, doc[0].ent_type_, doc[0].ent_kb_id_]
ent_ada_1 = [doc[1].text, doc[1].ent_type_, doc[1].ent_kb_id_]
ent_london_5 = [doc[5].text, doc[5].ent_type_, doc[5].ent_kb_id_]
print(ent_ada_0)  # ['Ada', 'PERSON', 'Q7259']
print(ent_ada_1)  # ['Lovelace', 'PERSON', 'Q7259']
print(ent_london_5)  # ['London', 'GPE', 'Q84']