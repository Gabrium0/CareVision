from pathlib import Path
import time
p = Path('C:\\Users\\Gabriel\\Downloads\\proj\\UsersGabrielAppDataLocalTempclaudept6\\test_hot_reloader_restarts_a_r0\\launches.txt')
with p.open('a', encoding='utf-8') as handle:
    handle.write('started\n')
time.sleep(30)
