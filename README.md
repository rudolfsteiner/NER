# Named Entity Recognition (NER)


Build several different models for named entity recognition (NER). NER is a subtask of information extraction that seeks to locate and classify named entities in text into pre-defined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc. For a given a word in a context, we want to predict whether it represents one of four categories:

• Person (PER): e.g. “Martha Stewart”, “Obama”, “Tim Wagner”, etc. Pronouns like “he” or “she” are
not considered named entities.

• Organization (ORG): e.g. “American Airlines”, “Goldman Sachs”, “Department of Defense”.

• Location (LOC): e.g. “Germany”, “Panama Strait”, “Brussels”, but not unnamed locations like “the
bar” or “the farm”.

• Miscellaneous (MISC): e.g. “Japanese”, “USD”, “1,000”, “Englishmen”.

We formulate this as a 5-class classification problem, using the four above classes and a null-class (O) for words that do not represent a named entity (most words fall into this category). For an entity that spans multiple words (“Department of Defense”), each word is separately tagged, and every contiguous sequence of non-null tags is considered to be an entity.

####1. A window into NER (window.py)

Build a simple baseline model that predicts a label for each token separately using features from a window around it.

####2. Recurrent neural nets for NER (rnn.py)

We will now tackle the task of NER by using a recurrent neural network (RNN).

####3. Grooving with GRUs (gru.py)

Using GRUs to reduce the problem of vanishing gradients and improve the performance of NER.
