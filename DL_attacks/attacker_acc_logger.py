import logging

class acc_logger:
    def __init__(self, Ctop, DL, log_file) -> None:
        
        self.Ctop = Ctop
        self.DL = DL
        # 创建一个日志记录器
        self.logger = logging.getLogger('acc_logger')
        self.logger.setLevel(logging.DEBUG)  # 设置日志级别
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # 设置处理器的日志级别

        # 创建一个日志格式器并将其添加到文件处理器中
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)

        # 将文件处理器添加到日志记录器中
        self.logger.addHandler(file_handler)
    
    def __call__(self, iter):
        self.logger.info(f'iter {iter}:')
        (loss_train, acc_train), (loss_test, acc_test), test_acc_lst, train_acc_lst, avg_result = self.DL.train_test_utility(drop_attacker=self.Ctop.active)
        test_result = ''
        train_result = '' 
        for i, test_acc in enumerate(test_acc_lst):
            test_result += f'{test_acc:.4f} '
        for i, train_acc in enumerate(train_acc_lst):
            train_result += f'{train_acc:.4f} '
        self.logger.info(train_result)
        self.logger.info(test_result)
        self.logger.info(avg_result)
        
        if hasattr(self.DL.attacker, 'result'):
            print(f'iter:{iter} Attack result: {str(self.DL.attacker.result)} {str(self.DL.attacker.mem_result)}')
            self.logger.info(f'Attack result:{str(self.DL.attacker.result)} {str(self.DL.attacker.mem_result)}')

