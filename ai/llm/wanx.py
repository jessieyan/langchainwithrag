#  Copyright (c) 2025 深圳优儿智能科技有限公司
#  All rights reserved.

import os
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import requests
from dashscope import ImageSynthesis

os.environ["DASHSCOPE_API_KEY"] = ""
def chat_wanx(user_message : str = None):
    rsp = ImageSynthesis.call(api_key=os.getenv("DASHSCOPE_API_KEY"),
                          model="wanx2.1-t2i-plus",
                          prompt=user_message,
                          n=1,
                          size='1024*1024')
    
    # Todo: need to point to the right path
    if rsp.status_code == HTTPStatus.OK:
        # 在当前目录下保存图片
        for result in rsp.output.results:
            file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
            with open('./%s' % file_name, 'wb+') as f:
                f.write(requests.get(result.url).content)
    else:
        print('sync_call Failed, status_code: %s, code: %s, message: %s' %
            (rsp.status_code, rsp.code, rsp.message))
        
    return rsp

    