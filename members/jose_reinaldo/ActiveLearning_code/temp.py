from ActiveLearning import ActiveLearning

# Dynamic parameters
dataset = 'conll'
model_opt = 'CNN-CNN-LSTM'
initial_set_seed = 2 # 0 2
query_budget = 4000
batch_size = 16

# Fixed parameters
confidence_threshold = 0.99
use_dev_set = False
labeled_percent_stop = 0.5
supervised_epochs = 50
query_fn = 'normalized_least_confidence'

# train_dic = ActiveLearning(
#     labeled_percent_stop = labeled_percent_stop,
#     supervised_epochs = supervised_epochs,
#     query_fn = query_fn,
#     dataset = dataset,
#     query_budget = query_budget,
#     model = model_opt,
#     initial_set_seed = initial_set_seed,
#     batch_size = batch_size,
#     early_stop_method = 'DUTE',
#     patience = 5,
#     use_dev_set = use_dev_set,
#     flag_self_label = True,
#     min_confidence = 0.99,
#     refinement_iter = 0
# )


from Supervised import Supervised
Supervised(
    supervised_epochs  = 50,
    dataset = 'conll',
    model = 'CNN-CNN-LSTM',
    batch_size = 16,
    use_dev_set = use_dev_set
    )