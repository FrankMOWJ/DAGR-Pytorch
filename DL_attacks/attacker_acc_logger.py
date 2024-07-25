import logging

class Attack_Accuracy_Logger:
    def __init__(self, Ctop, DL, log_file) -> None:
        
        self.Ctop = Ctop
        self.DL = DL
        # 创建一个日志记录器
        self.logger = logging.getLogger('my_logger')
        self.logger.setLevel(logging.DEBUG)  # 设置日志级别

        # # 创建一个文件处理器并设置文件路径
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
            
        # log_file = os.path.join(output_dir, log_name)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # 设置处理器的日志级别

        # 创建一个日志格式器并将其添加到文件处理器中
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 将文件处理器添加到日志记录器中
        self.logger.addHandler(file_handler)
    
    def __call__(self, iter):
        # suppose there is only 1 attacker
        print(f'iter:{iter} Attack result: {str(self.DL.attacker.result)}')
        self.logger.info(f'iter:{iter} {str(self.DL.attacker.result)}')
    

