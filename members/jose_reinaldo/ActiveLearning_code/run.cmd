python ActiveLearning.py ^
--train_path "dataset/conll03/eng_train.txt" --valid_path "dataset/conll03/eng_testa.txt" --test_path "dataset/conll03/eng_testb.txt" ^
--embedding_path "embedding/glove_conll03_100d.kv" --labeled_percent_stop 0.03 --epochs 50 ^
--save_training_path "saves/DevSetF1_training_save.pkl" --early_stop_method "DevSetLoss" ^
--model "CNN-biLSTM-CRF" --TokenSelfLabel_flag 0
