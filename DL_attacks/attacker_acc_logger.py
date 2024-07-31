import logging
import json
import signal 
import sys

class acc_logger:
    def __init__(self, Ctop, DL, log_file) -> None:
        
        self.Ctop = Ctop
        self.DL = DL
        
        self.log_path = log_file
        # 创建一个日志记录器
        self.logger = logging.getLogger('acc_logger')
        self.logger.setLevel(logging.DEBUG)  # 设置日志级别
        
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.DEBUG)  # 设置处理器的日志级别

        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        # console_handler.setFormatter(formatter)

        # 将文件处理器添加到日志记录器中
        self.logger.addHandler(file_handler)
        # self.logger.addHandler(console_handler)
        
        self.best_attack_result = {
            'iter': 0,
            'attack_metric': (0.0, 0.0, 0.0),
            'pred_distribution': (0, 0, 0, 0) 
        }
        
        signal.signal(signal.SIGINT, self.log_best_result)
    
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
        
        if hasattr(self.DL.attacker, 'result') and self.DL.attacker.result is not None:
            print(f'iter:{iter} Attack result: {str(self.DL.attacker.result)} {str(self.DL.attacker.mem_result)}')
            self.logger.info(f'Attack result:{str(self.DL.attacker.result)} {str(self.DL.attacker.mem_result)}')
            if self.DL.attacker.result[0] > self.best_attack_result['attack_metric'][0]:
                self.best_attack_result['iter'] = iter
                self.best_attack_result['attack_metric'] = self.DL.attacker.result
                self.best_attack_result['pred_distribution'] = self.DL.attacker.mem_result
    
    def log_best_result(self, sig, frame):
        self.logger.info('***** Best Attack Result ********')
        result_dict_str = json.dumps(self.best_attack_result, ensure_ascii=False, indent=4)
        self.logger.info(result_dict_str)
        
        print(f'log save at {self.log_path}')
        sys.exit(0)
        
