from qlib.config import REG_CN
from qlib.tests.data import GetData

provider = 'E:/qilb研究/.qlib_data/cn_data'
GetData().qlib_data(target_dir=provider, region=REG_CN)
