"""Module Build and Run logger for train"""
import logging
import os
import time


class TrainLogger:
    """Train logger class"""

    def __init__(self, cfg, root_dir):
        """Build logger

        Args:
            cfg:
                PROJECT:
                    EXP_NAME: name of experiment
                    root_dir: A string, dir root
        """
        self.logger = logging.getLogger()
        self.start_time = time.time()

        self.logger.setLevel(logging.INFO)

        dir_path = os.path.join(root_dir, cfg.project.name)
        self.create_directory(dir_path)
        localtime = time.localtime(self.start_time)
        nowdate = f"{localtime.tm_year}_{localtime.tm_mon}_{localtime.tm_mday}"
        nowtime = f"{localtime.tm_hour}:{localtime.tm_min}:{localtime.tm_sec}"
        self.log_file = os.path.join(
            dir_path, f"{cfg.project.name}_{nowdate}_{nowtime}.log"
        )

        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)

        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def print_log(self, message):
        """print log with logger

        Args:
            message: Log message
        """
        self.logger.info(message)

    @staticmethod
    def create_directory(directory):
        """make dir if not exist

        Args:
            directory: dir path
        """
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError:
            print("Error: Failed to create the directory")
