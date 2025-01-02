import os
import logging
from datetime import datetime


LOG_FILE_DIR="logs"
LOG_FILE_NAME=f"{datetime.now().strftime('%d-%B-%Y_%H_%M_%S')}.log"
LOG_FILE_PATH=os.path.join(os.getcwd(),LOG_FILE_DIR,LOG_FILE_NAME)

os.makedirs(LOG_FILE_DIR,exist_ok=True)

logging.basicConfig(filename=LOG_FILE_PATH,
                    format='%(asctime)s-%(filename)s-%(lineno)d-%(levelname)s-%(message)s',
                    level=logging.INFO,
                    datefmt='%d-%B-%Y %H:%M:%S'
                    )

# if __name__=='__main__':
#     logging.info('Logging has been started')