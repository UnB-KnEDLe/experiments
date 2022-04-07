from metaflow import FlowSpec, step, Parameter

class Data_Processing_Flow(FlowSpec):
    """
        The flow performs the following steps:
        1) Start flow
        2) Load labeled datasets, tokenization labels and filter REGEX patterns
        3) Process data, separating texts, labels and their location, and filter patterns
        4) Mark the text based on the labeled entities' locations
        5) Filter the labeled texts using REGEX patterns
        6) Tokenize filtered texts (word_tokenize and IOB1 format)
        7) Save tokenized texts in JSON
        8) End flow
    """

    # Opções default feitas para processamento de dados base ouro, em extratos de contrato
    # Arquivos base de entrada: Labeled Acts, Labeled Entities, JSON com tokens, JSON com regras REGEX de filtragem

    labeled_acts_path = Parameter('labeled_acts_path',
                                    help='Path to Labeled Acts dataset')

    labeled_entities_path = Parameter('labeled_entities_path',
                                        help='Path to Labeled Entities dataset')

    filter_act = Parameter('filter_act',
                                help="Type of act whose texts must be extracted",
                                default="extrato_de_contrato_ou_convenio")

    tokens_eq_rep_path = Parameter('tokens_eq_rep_path',
                                help='Replacement rules defined upon the entities')

    eq_column = Parameter('eq_column',
                            help='Column where the token replacement must be done',
                            default='title_ent')

    filter_patterns_path = Parameter('filter_patterns_path',
                                    help='Filter REGEX patterns, with patterns and rep keys, where in the first the list of the patterns goes and in the second how to replace')
    
    save_path = Parameter('save_path',
                            help='Path where data will be stored')

    @step
    def start(self):
        from processing import JSON

        print("flow started")

        self.next(self.load_data)

    @step
    def load_data(self):
        import pandas as pd

        self.tokens_eq_rep = JSON.read_json(self.tokens_eq_rep_path)
        self.filter_patterns = JSON.read_json(self.filter_patterns_path)

        self.labeled_acts = pd.read_parquet(self.labeled_acts_path)
        self.labeled_entities = pd.read_parquet(self.labeled_entities_path)

        self.next(self.process_data)

    @step
    def process_data(self):
        # Processamento dos dados do dataset: https://github.com/UnB-KnEDLe/datasets/blob/master/anotacoes_atos_de_contrato_e_licitacao.md
        # Texto no dataset de Labeled Acts e informações de entidades rotuladas em Labeled Entities

        from processing import Processing

        acts_tuples = Processing.preprocess_dataset(self.labeled_acts, self.labeled_entities, self.filter_act, self.eq_column, self.tokens_eq_rep)
        self.data_dict = {
            "texts": acts_tuples,
            "token_equivalences": self.tokens_eq_rep,
            "filter_patterns": self.filter_patterns
        }

        self.next(self.mark_text)

    @step
    def mark_text(self):
        # A anotação foi feita em dado não filtrado, logo é preciso marcar as entidades rotuladas antes da filtragem
        # Sẽ essa marcação não for feita, perde-se o endereço das entidades
        # Isso também permite a tokenização correta, já que força espaçamento entre os tokens

        from processing import Processing

        self.treated_texts = []

        for text in self.data_dict["texts"]:
            self.treated_texts.append(Processing.mark_text(text=text[0], loc_tokens=text[1]))

        self.next(self.filter_text)

    @step
    def filter_text(self):
        from processing import Processing

        cleaned_texts = []

        for text in self.treated_texts:
            for i in range(len(self.data_dict["filter_patterns"]["patterns"])):
                cleaned_texts.append(Processing.clean_text(text=text, filter_patterns=self.data_dict["filter_patterns"]["patterns"][i], rep=self.data_dict["filter_patterns"]["rep"][i]))

        self.treated_texts = cleaned_texts
        self.next(self.tokenize_text)

    @step
    def tokenize_text(self):
        # Tokenização feita com word_tokenize do nltk
        # Formato IOB1

        from processing import Processing

        self.iob_texts = []

        for text in self.treated_texts:
            self.iob_texts.append(Processing.tokenize_text(text=text))

        self.next(self.save_data)

    @step
    def save_data(self):
        import os
        from processing import JSON

        iob_texts_dict = {"iob_texts": ["\n".join(t) for t in self.iob_texts]}
        JSON.write_json(iob_texts_dict, os.path.join(self.save_path, "iob_texts.json"))

        self.next(self.end)

    @step
    def end(self):
        print("flow finished")

if __name__ == '__main__':
    Data_Processing_Flow()