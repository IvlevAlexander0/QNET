from utils import clean_log_folders, find_log_folders
from datetime import datetime

folders = find_log_folders(
    log_root='D:\\WORK\\Severstal\\MAIN_REP\\logs',
#    model_name='resnet18',
#    model_mode='quantum',
    min_epochs=10,
    after_date='20261101'
)
print(folders)


# Проверить, что будет удалено
#clean_log_folders('D:/WORK/Severstal/MAIN_REP/logs', min_epochs=3, dry_run=False)