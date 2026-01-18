from data.loader import FileIO


class SELFRec(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config
        # data  = [[user_id, item_id, weight]]
        self.training_data = FileIO.load_data_set(config['training.set'], config['model.type'])  # 三元组
        self.test_data = FileIO.load_data_set(config['test.set'], config['model.type'])
        self.valid_data = FileIO.load_data_set(config['valid.set'], config['model.type'])

        self.kwargs = {}
        if config.contain('social.data'):
            social_data = FileIO.load_social_data(self.config['social.data'])
            self.kwargs['social.data'] = social_data
        print('Reading data and preprocessing...')

    def execute(self):
        import importlib
        module_path = 'model.' + self.config['model.type'] + '.' + self.config['model.name']
        module = importlib.import_module(module_path)
        model_class = getattr(module, self.config['model.name'])
        recommender = model_class(self.config, self.training_data, self.test_data, self.valid_data, **self.kwargs)
        recommender.execute()
