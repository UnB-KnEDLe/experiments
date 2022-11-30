# Tutorial de utilização da extração de entidades de um arquivo JSON

This tutorial is meant to help the process of extracting acts from the section 3 of the DODF JSON files. These acts are the of the following type:

- Contrato or Convênio
- Aditamento
- Licitação
- Anulação / Revogação
- Suspensão


### Importação da biblioteca DODFMiner

You might import the DODFMiner library in order to extract the acts from a JSON file. You can do that by doing this import:

```Python
from dodfminer.extract.polished.core import ActsExtractor
```

Each of the 5 types of acts have their own class that manages the whole process of extraction from the JSON file, but it is possible to extract all of them at once. To do that, you have to use the ActsExtractor method.

```Python
ActsExtractor.get_all_obj_sec3(file)
```

- Parameters:
    - **file** (string) - Path to JSON file.

- Returns:
    - Dictionary containing the class objects correspondent to each type of act.

Within each class object in the returned dictionary, there is a dataframe containing all the information about each act of that type found in the JSON.

The DataFrame information follows the pattern:

- numero_dodf
- titulo
- text
- [The respective entities of each type of act]
