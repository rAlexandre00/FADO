class Dataset:

    def __init__(self, train_data=None, test_data=None, target_test_data=None):
        """

            Parameters:
                train_data (dict): with x and y keys
                test_data (dict): with x and y keys
                target_test_data (dict): with x and y keys
        """
        if train_data:
            assert list(train_data.keys()) == ['x', 'y']
        if test_data:
            assert list(test_data.keys()) == ['x', 'y']
        if target_test_data:
            assert list(target_test_data.keys()) == ['x', 'y']

        self.train_data = train_data
        self.test_data = test_data
        self.target_test_data = target_test_data

    def has_target_test(self):
        return self.target_test_data is not None
