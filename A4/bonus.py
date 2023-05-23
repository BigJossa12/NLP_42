from ufal.udpipe import Model, Pipeline

def parse_text(text, model_path):
    # load the model from the model file
    model = Model.load(model_path)
    if not model:
        print("Cannot load model from file '%s'" % model_path)
        return None

    # create a pipeline for processing text
    pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

    # process the text and return the parsed output
    return pipeline.process(text)

# models can be downloaded from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131
english_model_path = 'english-partut-ud-2.5-191206.udpipe'
russian_model_path = 'russian-syntagrus-ud-2.5-191206.udpipe'

# English and Russian sentences to parse
english_sentences = ["I saw the man on the hill with a telescope.",
                     "Leaving the store, the sun was shining brightly.",
                     "I have nearly finished the book.",
                     "Would you like coffee or tea and cookies?"]
russian_sentences = ["Я видел мужчину на холме с телескопом.",
                     "Покидая магазин, солнце ярко светило.",
                     "Я почти закончил книгу.",
                     "Вы бы хотели кофе или чай и печенье?"]

# parse the English sentences
for sentence in english_sentences:
    print(parse_text(sentence, english_model_path))

# parse the Russian sentences
for sentence in russian_sentences:
    print(parse_text(sentence, russian_model_path))
