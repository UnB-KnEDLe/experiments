from metaflow import FlowSpec, step, IncludeFile, batch, S3, Parameter, current

class BiLSTM_CRF_Flow(FlowSpec):
    """
        The flow represents the following steps
        1) Load the labeled data stored in a .txt file
        2) Split the data in training, validation and testing sets
        3) Process the data
        4) Train and save the BiLSTM-CRF model
        5) Load a trained model
        6) Predict input data
    """

    labeled_data_path = Parameter('labeled_data_path',
    								help='.parquet DataFrame with labeled data')

    num_epochs = Parameter('num_epochs',
    						help='Number of epochs to fit the model',
    						default=30)

    lrate = Parameter('lrate',
    					help='Learning rate to use in the model',
    					default=0.01)

    save_path = Parameter('save_path',
    						help='Path where to save data')

    model_path = Parameter('model_path',
    						help='Path to stored model')

    idx_folder = Parameter('idx_folder',
    						help='Path of the folder where word2idx and tag2idx are stored')

    project_name = Parameter('project_name',
    						 help='Name of the project')

    texts_predict_path = Parameter('texts_predict_path',
    								help='Path to .txt file with the texts to be predicted')

    @step
    def start(self):
        """
            Load labeled data
        """
        with open(self.labeled_data_path) as file:
        	self.labeled_data = file.readlines()
        file.close()

        self.next(self.split_data)

    @step
    def split_data(self):
        """
            Split labeled data in training, testing and validation sets
        """
        from process_data import preprocess
        from sklearn.model_selection import train_test_split
        
        self.X, self.Y = preprocess(self.labeled_data)
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        self.X_test, self.X_valid, self.Y_test, self.Y_valid = train_test_split(self.X_test, self.Y_test, test_size=0.5, random_state=42)
        self.next(self.process_data)

    @step
    def process_data(self):
        """
            Process splitted data, creating the ids for the words and tags used in the labeled data
        """
	    from process_data import word_dict, tag_dict, numericalize, add_special_tokens
	    
	    self.word2idx = word_dict(self.X_train)
	    self.tag2idx  = tag_dict(self.Y_train)
	    
	    self.X_train, self.Y_train = numericalize(self.X_train, self.word2idx, self.Y_train, self.tag2idx)
	    self.X_valid, self.Y_valid = numericalize(self.X_valid, self.word2idx, self.Y_valid, self.tag2idx)
	    self.X_test, self.Y_test = numericalize(self.X_test, self.word2idx, self.Y_test, self.tag2idx)
	    
	    self.X_train, self.Y_train = add_special_tokens(self.X_train, self.word2idx, self.Y_train, self.tag2idx)
	    self.X_valid, self.Y_valid = add_special_tokens(self.X_valid, self.word2idx, self.Y_valid, self.tag2idx)
	    self.X_test, self.Y_test = add_special_tokens(self.X_test, self.word2idx, self.Y_test, self.tag2idx)
	    self.next(self.train_model)

    @step
    def train_model(self):
        """
            Train BiLSTM-CRF model
        """
        import os
        import torch
        from bilstm_crf import bilstm_crf, fit_model
        from process_data import write_pkl

        self.momentum=0.9
        self.model = bilstm_crf(word2idx=self.word2idx, tag2idx=self.tag2idx)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.model, self.f1_history, self.mean_loss_history = fit_model(self.model, self.device, self.X_train, self.Y_train, self.X_test, self.Y_test, self.X_valid, self.Y_valid, self.word2idx, self.tag2idx, self.num_epochs, self.lrate, self.momentum)

        torch.save(self.model.state_dict(), os.path.join(self.save_path, "{}-bilstm_crf_model-{}_epochs".format(self.project_name, self.num_epochs)))
        write_pkl(self.word2idx, os.path.join(self.save_path, "{}_word2idx.pickle".format(self.project_name)))
        write_pkl(self.tag2idx, os.path.join(self.save_path, "{}_tag2idx.pickle".format(self.project_name)))
        self.next(self.load_model)

    @step
    def load_model(self):
        """
            Load stored BiLSTM-CRF model
        """
    	import os
    	import torch
    	from process_data import read_pkl
    	from bilstm_crf import bilstm_crf

    	self.word2idx = read_pkl(os.path.join(self.idx_folder, "{}_word2idx.pickle".format(self.project_name)))
    	self.tag2idx = read_pkl(os.path.join(self.idx_folder, "{}_tag2idx.pickle".format(self.project_name)))

    	self.model = bilstm_crf(word2idx=self.word2idx, tag2idx=self.tag2idx)
    	self.loaded_model = torch.load(self.model_path, map_location='cpu')
    	self.model.load_state_dict(self.loaded_model)
    	self.model = self.model.to('cpu')
    	self.next(self.deploy)

    @step
    def deploy(self):
        """
            Deploy the loaded model and use it to predict input texts stored in .txt files
        """
    	import os
    	from process_data import IOBify
    	from bilstm_crf import predict_text

    	self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}

    	text_file = open(self.texts_predict_path, "r")
    	self.texts_predict = text_file.read()
    	text_file.close()

    	self.texts_predict = self.texts_predict.split("\n\n")
    	self.tagged_texts = []

    	for base_text in self.texts_predict:
    		pred_text, pred_tags = predict_text(base_text, self.word2idx, self.idx2tag, self.model)

    		tagged_text = []
    		for i in range(len(pred_text)):
    			tagged_text.append(pred_text[i] + " X X " + pred_tags[i])
    		self.tagged_texts.append(tagged_text)

    	with open(os.path.join(self.save_path, "{}_predicted_texts.txt".format(self.project_name)), "w") as f:
    		f.write("\n\n".join(["\n".join(t) for t in self.tagged_texts]))
    	self.next(self.end)

    @step
    def end(self):
        """
            End of flow
        """

        print("finished")

if __name__ == '__main__':
    BiLSTM_CRF_Flow()