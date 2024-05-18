import logging 
import os
import datetime
def get_logger(path=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = logging.getLogger(__name__)
    if path==None:
        log_folder = f"logs/{timestamp}"
    else:
        log_folder = f"{path}/{timestamp}"
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "log.txt")
    fromat2='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handler = logging.FileHandler(log_file)
    console = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format=fromat2,
        handlers=[handler, console]
    )
    return logger

exported_logger =get_logger()
