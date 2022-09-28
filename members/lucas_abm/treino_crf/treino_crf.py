from sklearn.model_selection import train_test_split, KFold
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score
from seqeval.metrics import classification_report, f1_score
import sklearn_crfsuite
import pandas as pd
import nltk
import numpy as np

class CRF_Flow():
  X_train = None
  X_test = None
  y_train = None
  y_test = None
  X = None
  model = None

  f1_score = 0
  result = ""

  def __init__(self, csv_path : str = None, c1 = 0.17, c2 = 0.17, n_iterations : int = 50, cross_validation = False):
    """
    Construtor.

    Caso o "csv_path" seja incluído, o construtor chamará a função `self.run_all`
    """
    self.c1 = c1
    self.c2 = c2
    self.n_iterations = n_iterations
    # nltk.download('punkt')

    if csv_path is not None:
      self.run_all(csv_path, cross_validation=cross_validation)

  def init_architeture(self):
    self.model = sklearn_crfsuite.CRF(
      algorithm = 'lbfgs',
      c1=self.c1,
      c2=self.c2,
      max_iterations=self.n_iterations,
      all_possible_transitions=True
    )

  def run_all(self, csv_path : str, cross_validation = False) -> None:
    """
      Função para rodar todo o fluxo de uma vez:

      - Carregar dados
      - Treinar modelo
      - Validar modelo
      - Salvar resultados em um txt

      Esse método criará uma pasta chamada `results`
    """
    self.load(csv_path, split = not cross_validation)

    if cross_validation is True:
      self.cross_validation()

    else:
      self.init_architeture()
      self.train()
      self.validation()

    self.save("results/" + csv_path.split("/")[-1].split(".")[0] + ".txt")

  def load(self, csv_path : str, split = True) -> None:
    """Input: path para o arquivo csv contendo os dados

    carrega os arquivos IOB e
    divide em treino e teste (80% treino e 20% teste)
    """
    df = pd.read_csv(csv_path)
    X = df["treated_text"]
    y = df["IOB"]

    # TOKENIZER = nltk.RegexpTokenizer(r"\w+").tokenize
    X = X.apply(nltk.tokenize.word_tokenize)

    y = y.apply(lambda y: y.split()).values
    X = X.apply(self.__get_features).values

    if split is True:
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    else:
      self.X_train = X
      self.y_train = y

  def train(self) -> None:
    """
    Treina o modelo com os dados de treino guardados.
    """
    self.model.fit(self.X_train, self.y_train)

  def cross_validation(self) -> None:
    """
    Valida o modelo fazendo a validação cruzada.

    Dados gerados:

    - `self.f1`: inteiro com o f1 score do modelo
    - `self.result`: string com o classification report do modelo
    """

    predictions = np.array([], dtype="object")
    true = np.array([], dtype="object")
    tenfold = KFold(n_splits=10, shuffle=True, random_state=1)
    i = 0
    for train_index, test_index in tenfold.split(self.X_train):
      print(f"Treino {i+1}/10")
      self.init_architeture()
      self.model.fit(self.X_train[train_index], self.y_train[train_index])

      classes = list(self.model.classes_)
      classes.remove('O')
      predictions = np.concatenate([predictions, self.model.predict(self.X_train[test_index])], dtype="object")
      true = np.concatenate([true, self.y_train[test_index]], dtype="object")
      i+=1
    # print(true[0:10])
    # print(predictions[0:10])
    self.result = classification_report(true, predictions)
    self.f1 = f1_score(true, predictions)

  def validation(self) -> None:
    """
    Valida o modelo fazendo a predição com base nos dados de teste.

    Dados gerados:

    - `self.f1`: inteiro com o f1 score do modelo
    - `self.result`: string com o classification report do modelo
    """
    classes = list(self.model.classes_)
    classes.remove('O')

    y_pred = self.model.predict(self.X_test)

    self.f1 = flat_f1_score(self.y_test, y_pred, average='weighted', labels=classes)
    self.result = flat_classification_report(self.y_test, y_pred, labels=classes)
    return self.result

  def save(self, path : str) -> None:
    """
    Salva classification report no caminho especificado
    """
    f = open(path, "w")
    f.write(self.result)
    f.close()

  def __get_features(self, sentence):
      """Create features for each word in act.
      Create a list of dict of words features to be used in the predictor module.
      Args:
        act (list): List of words in an act.
      Returns:
        A list with a dictionary of features for each of the words.
      """
      sent_features = []
      for i in range(len(sentence)):
        # print(sentence[i])
        word_feat = {
          # Palavra atual
          'word': sentence[i].lower(),
          'capital_letter': sentence[i][0].isupper(),
          'all_capital': sentence[i].isupper(),
          'isdigit': sentence[i].isdigit(),
          # Uma palavra antes
          'word_before': '' if i == 0 else sentence[i-1].lower(),
          'word_before_isdigit': '' if i == 0 else sentence[i-1].isdigit(),
          'word_before_isupper': '' if i == 0 else sentence[i-1].isupper(),
          'word_before_istitle': '' if i == 0 else sentence[i-1].istitle(),

          # Uma palavra depois
          'word_after': '' if i+1 >= len(sentence) else sentence[i+1].lower(),
          'word_after_isdigit': '' if i+1 >= len(sentence) else sentence[i+1].isdigit(),
          'word_after_isupper': '' if i+1 >= len(sentence) else sentence[i+1].isupper(),
          'word_after_istitle': '' if i+1 >= len(sentence) else sentence[i+1].istitle(),

          'BOS': i == 0,
          'EOS': i == len(sentence)-1
        }
        sent_features.append(word_feat)
      return sent_features