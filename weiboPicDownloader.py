# -*- coding: utf-8 -*-

import argparse
import concurrent.futures
import datetime
import json
import locale
import math
import operator
import os
import platform
import re
import sys
import time
from functools import reduce

import random
import requests
import threading

# 全局 Session，自动管理 Cookie（线程安全）
_session = requests.Session()
_session.headers.update({
    "referer": "https://m.weibo.cn/",
    "User-Agent": "Mozilla/5.0 (Linux; Android 9; Pixel 3 XL) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.80 Mobile Safari/537.36",
})
_session.verify = False
_session_lock = threading.Lock()

# 反爬智能延迟器
_anti_scrape = {
    "consecutive_errors": 0,    # 连续错误计数
    "base_delay": 1.0,          # 基础延迟（秒）
    "max_delay": 30.0,          # 最大延迟（秒）
    "lock": threading.Lock(),
}


def anti_scrape_sleep(is_api=False):
    """
    智能延迟：正常请求随机抖动，连续出错时指数退避。

    参数:
    - is_api (bool): 是否为翻页 API 请求（翻页比下载更容易触发反爬）
    """
    with _anti_scrape["lock"]:
        errors = _anti_scrape["consecutive_errors"]
        base = _anti_scrape["base_delay"]
        cap = _anti_scrape["max_delay"]

    if errors > 0:
        # 指数退避：2^errors * base，带上限和随机抖动
        delay = min(cap, base * (2 ** errors))
        delay *= random.uniform(0.8, 1.2)
    else:
        # 正常请求：基础间隔 + 随机抖动
        delay = base * random.uniform(0.5, 1.5)

    if is_api:
        # API 翻页请求额外加一点缓冲
        delay += random.uniform(0.5, 1.5)

    time.sleep(delay)


def anti_scrape_success():
    """请求成功时重置连续错误计数"""
    with _anti_scrape["lock"]:
        _anti_scrape["consecutive_errors"] = 0


def anti_scrape_fail():
    """请求失败时递增连续错误计数"""
    with _anti_scrape["lock"]:
        _anti_scrape["consecutive_errors"] += 1
        count = _anti_scrape["consecutive_errors"]
    print_fit("anti-scrape: consecutive errors={}, backing off".format(count))
    return count

"""这段Python代码主要用于设置命令行参数解析器，以便用户可以通过命令行参数来配置微博图片下载器的行为。具体功能包括：
系统兼容性和编码设置：尝试设置系统默认编码为UTF-8，并检查是否在Windows系统上运行，如果是，则进行一些特定的初始化操作。
禁用不安全请求警告：禁用urllib3库的不安全请求警告。
命令行参数解析：使用argparse库定义了一系列命令行参数，包括用户昵称或ID、文件路径、保存目录、线程池大小、重试次数、请求间隔、Cookie、命名格式、是否下载视频等。"""
try:
    reload(sys)
    sys.setdefaultencoding("utf8")
except:
    pass

is_python2 = sys.version[0] == "2"
system_encoding = sys.stdin.encoding or locale.getpreferredencoding(True)

if platform.system() == "Windows":
    if operator.ge(
            *map(
                lambda version: list(map(int, version.split("."))),
                [platform.version(), "10.0.14393"],
            )
    ):
        os.system("")
    else:
        import colorama

        colorama.init()

try:
    requests.packages.urllib3.disable_warnings(
        requests.packages.urllib3.exceptions.InsecureRequestWarning
    )
except:
    pass

parser = argparse.ArgumentParser(prog="weiboPicDownloader")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "-u",
    metavar="user",
    dest="users",
    nargs="+",
    help="specify nickname or id of weibo users",
)
group.add_argument(
    "-f",
    metavar="file",
    dest="files",
    nargs="+",
    help="import list of users from files",
)
parser.add_argument(
    "-d", metavar="directory", dest="directory", help="set picture saving path"
)
parser.add_argument(
    "-y",
    dest="yes",
    action="store_true",
    help="skip confirmation and create pictures directory",
    default=False,
)
parser.add_argument(
    "-s",
    metavar="size",
    dest="size",
    default=20,
    type=int,
    help="set size of thread pool",
)
parser.add_argument(
    "-r",
    metavar="retry",
    dest="retry",
    default=2,
    type=int,
    help="set maximum number of retries",
)
parser.add_argument(
    "-i",
    metavar="interval",
    dest="interval",
    default=1,
    type=float,
    help="set interval for feed requests",
)
parser.add_argument("-c", metavar="cookie", dest="cookie", help="set cookie if needed")
parser.add_argument(
    "-b",
    metavar="boundary",
    dest="boundary",
    default=":",
    help="focus on weibos in the id range",
)
parser.add_argument(
    "-n", metavar="name", dest="name", default="{name}", help="customize naming format"
)
parser.add_argument(
    "-v", dest="video", action="store_true", help="download videos together"
)
parser.add_argument(
    "-o", dest="overwrite", action="store_true", help="overwrite existing files"
)
parser.add_argument("--cookie-all", metavar="cookieall", dest="cookieall", help="set cookie all if needed")

def nargs_fit(parser, args):
    """
    根据参数解析器和传入的参数列表，调整参数列表以适应nargs规范。

    此函数旨在处理短参数和长参数，并根据参数解析器的规范调整参数列表，
    以确保正确解析具有不同nargs值的短参数（单破折号）和长参数（双破折号）。

    参数:
    - parser: 参数解析器对象，应包含 `_option_string_actions` 属性，用于获取所有可选参数的信息。
    - args: 参数列表，通常是从命令行接收的参数。

    返回:
    - 调整后的参数列表，以适应nargs规范。
    """
    flags = parser._option_string_actions
    short_flags = [flag for flag in flags.keys() if len(flag) == 2]
    long_flags = [flag for flag in flags.keys() if len(flag) > 2]
    short_flags_with_nargs = set([flag[1] for flag in short_flags if flags[flag].nargs])
    short_flags_without_args = set(
        [flag[1] for flag in short_flags if flags[flag].nargs == 0]
    )
    validate = lambda part: (
                                    re.match(r"-[^-]", part)
                                    and (
                                            set(part[1:-1]).issubset(short_flags_without_args)
                                            and "-" + part[-1] in short_flags
                                    )
                            ) or (part.startswith("--") and part in long_flags)

    greedy = False
    for index, arg in enumerate(args):
        if arg.startswith("-"):
            valid = validate(arg)
            if valid and arg[-1] in short_flags_with_nargs:
                greedy = True
            elif valid:
                greedy = False
            elif greedy:
                args[index] = " " + args[index]
    return args


def print_fit(string, pin=False):
    """
    根据环境和参数选择性地打印字符串。

    该函数旨在适应不同情况下的输出需求，包括在Python 2环境中对字符串进行编码，
    以及根据pin参数决定是否将输出固定在一行。

    参数:
    - string: 待打印的字符串。
    - pin: 布尔值，指示是否应将输出固定在一行（默认为False）。
    """
    if is_python2:
        string = string.encode(system_encoding)
    if pin:
        sys.stdout.write("\r\033[K")
        sys.stdout.write(string)
        sys.stdout.flush()
    else:
        sys.stdout.write(string + "\n")


def input_fit(string=""):
    """
    根据Python版本选择合适的输入函数。

    在Python 2和Python 3中，输入函数的名称不同：Python 2使用raw_input，而Python 3使用input。
    本函数根据Python解释器的版本，选择合适的输入函数，并处理字符编码问题，以确保在不同的Python版本下
    能够正确读取用户输入。

    参数:
    string (str): 提示信息字符串，显示给用户。

    返回:
    str: 用户的输入。
    """
    if is_python2:
        return raw_input(string.encode(system_encoding)).decode(system_encoding)
    else:
        return input(string)


def merge(*dicts):
    """
    合并多个字典为一个字典。

    这个函数接受多个字典作为输入参数，并将它们合并成一个字典。如果存在相同的键，
    后续字典中的值会覆盖前面字典中的值。

    参数:
    *dicts: 可变数量的字典参数。允许合并两个或多个字典。

    返回:
    一个字典，包含所有输入字典中的键值对。
    """
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def quit(string=""):
    """
    打印给定的字符串并退出程序。

    参数:
    string (str): 要打印的字符串，默认为空字符串。

    返回:
    无返回值。
    """
    print_fit(string)
    exit()


def make_dir(path):
    """
    创建目录

    该函数尝试创建一个目录，如果遇到任何异常（例如目录已存在、权限不足等），
    则退出程序并输出异常信息。使用此函数需谨慎，因为它会导致程序立即终止。

    参数:
    path (str): 要创建的目录路径

    返回:
    无
    """
    try:
        os.makedirs(path)
    except Exception as e:
        quit(str(e))


def confirm(message):
    """
    请求用户确认以获取布尔值作为响应。

    该函数通过向用户显示一条消息并等待用户输入来请求确认。它接受用户的
    'Y'或'y'作为真值，'N'或'n'作为假值。如果用户输入的不是预期的值，函数
    将继续提示直到得到有效的输入。

    参数:
    message (str): 显示给用户的提示消息。

    返回:
    bool: 如果用户输入'Y'或'y'则返回True，如果用户输入'N'或'n'则返回False。
    """
    while True:
        answer = input_fit("{} [Y/n] ".format(message)).strip()
        if answer == "y" or answer == "Y":
            return True
        elif answer == "n" or answer == "N":
            return False
        print_fit("unexpected answer")


def progress(part, whole, percent=False):
    """
    根据部分数量和总量计算进度。

    参数:
    part (int): 完成的部分数量。
    whole (int): 总量。
    percent (bool, optional): 是否显示百分比。默认为False。

    返回:
    str: 如果percent为True，则返回格式为'部分/总量(百分比%)'的字符串；
         否则，返回格式为'部分/总量'的字符串。
    """
    if percent:
        return "{}/{}({}%)".format(part, whole, (float(part) / whole * 100))
    else:
        return "{}/{}".format(part, whole)


def request_fit(method, url, max_retry=0, cookie=None, stream=False, is_api=False):
    """
    发起HTTP请求并进行重试，内置反爬检测。

    使用全局 Session 自动管理 Cookie：服务端返回的 Set-Cookie 会自动
    保存在 Session 中，后续请求会自动携带，无需手动拼接。

    内置反爬机制：
    - 检测 418 状态码（微博反爬经典返回码）
    - 检测非 200 状态码
    - 成功时重置退避计数，失败时递增退避计数
    - 通过 anti_scrape_sleep() 在请求前自动延迟

    参数:
    - method (str): HTTP方法，例如GET、POST等。
    - url (str): 请求的URL地址。
    - max_retry (int): 最大重试次数，默认为0，即不重试。
    - cookie (str): 初始请求的Cookie值（可选，注入到 Session 后由 Session 自动管理）。
    - stream (bool): 是否以流的形式读取响应，默认为False。
    - is_api (bool): 是否为翻页 API 请求（会额外增加延迟缓冲）。

    返回:
    - requests.Response: 请求的响应对象。
    """
    with _session_lock:
        if cookie and "Cookie" not in _session.headers:
            _session.headers["Cookie"] = cookie

    anti_scrape_sleep(is_api=is_api)

    response = _session.request(method, url, timeout=5, stream=stream)

    if response.status_code == 418:
        anti_scrape_fail()
        print_fit("WARNING: HTTP 418 detected, anti-scraping triggered! backing off...")
    elif response.status_code != 200:
        anti_scrape_fail()
    else:
        anti_scrape_success()

    return response


def read_from_file(path):
    """
    从指定的文件路径读取内容并返回。

    该函数尝试打开并读取指定路径下的文件内容。如果是在Python 2环境中运行，
    它会将每行的内容进行解码。在Python 3环境中，则直接读取。这是为了确保
    函数在不同的Python版本中能够正确处理文件内容。

    参数:
    path (str): 文件的路径。

    返回:
    list: 包含文件每行内容的列表，每行内容都已去除换行符和可能的空白字符。
    """
    try:
        with open(path, "r") as f:
            return [
                line.strip().decode(system_encoding) if is_python2 else line.strip()
                for line in f
            ]
    except Exception as e:
        quit(str(e))


def nickname_to_uid(nickname):
    """
    根据微博昵称获取用户UID。

    通过发送GET请求到微博移动端URL，转换昵称到UID。
    兼容以下三种重定向场景：
    1. 最终URL以 /u/XXXXXXXXXX 结尾（正常情况）
    2. 重定向历史中某一跳的URL以 /u/XXXXXXXXXX 结尾
    3. 被跳转到 passport 域名时，UID 编码在 url 参数里（如 %2Fu%2F5026377162）

    参数:
    nickname (str): 微博用户的昵称。

    返回:
    str: 用户的UID，如果找不到则返回None。
    """
    url = "https://m.weibo.cn/n/{}".format(nickname)
    response = request_fit("GET", url)

    # 场景1：最终 URL 直接匹配（原始逻辑）
    if re.search(r"/u/\d{10}$", response.url):
        return response.url[-10:]

    # 场景2：从重定向历史中提取
    for r in response.history:
        if re.search(r"/u/\d{10}$", r.url):
            return r.url[-10:]
        loc = r.headers.get("Location", "")
        m = re.search(r"/u/(\d{10})", loc)
        if m:
            return m.group(1)

    # 场景3：UID 以 URL 编码形式藏在 passport 跳转参数里
    m = re.search(r"%2Fu%2F(\d{10})", response.url)
    if m:
        return m.group(1)

    return


def uid_to_nickname(uid):
    """
    根据用户ID获取昵称

    Args:
    uid (str): 用户ID

    Returns:
    str: 用户昵称，如果获取失败则返回None
    """
    url = "https://m.weibo.cn/api/container/getIndex?type=uid&value={}".format(uid)
    response = request_fit("GET", url)
    try:
        return json.loads(response.text)["data"]["userInfo"]["screen_name"]
    except:
        return


def bid_to_mid(string):
    """
    将bid编码字符串转换为mid（消息ID）。

    bid编码是一种自定义的Base62编码方式，它使用62个字符（0-9, a-z, A-Z）来表示信息。
    本函数的目的是将这种编码的字符串转换回其原始的整数形式（mid）。

    参数:
    string (str): 需要转换的bid编码字符串。

    返回:
    int: 转换后的mid（消息ID）。
    """
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alphabet = {x: n for n, x in enumerate(alphabet)}

    splited = [
        string[(g + 1) * -4: g * -4 if g * -4 else None]
        for g in reversed(range(math.ceil(len(string) / 4.0)))
    ]
    convert = lambda s: str(
        sum([alphabet[c] * (len(alphabet) ** k) for k, c in enumerate(reversed(s))])
    ).zfill(7)
    return int("".join(map(convert, splited)))


def parse_date(text):
    """
    解析日期字符串并返回相应的日期。

    参数:
    text (str): 日期或时间描述的字符串。

    返回:
    date: 解析后的日期对象。
    """
    now = datetime.datetime.now()
    if "前" in text:
        if "小时" in text:
            return (
                    now - datetime.timedelta(hours=int(re.search(r"\d+", text).group()))
            ).date()
        else:
            return now.date()
    elif "昨天" in text:
        return now.date() - datetime.timedelta(days=1)
    elif re.search(r"^[\d|-]+$", text):
        return datetime.datetime.strptime(
            ((str(now.year) + "-") if not re.search(r"^\d{4}", text) else "") + text,
            "%Y-%m-%d",
        ).date()


def compare(standard, operation, candidate):
    """
    Compare the given standard value with a set of candidate values based on the specified operation.

    Parameters:
    standard (int/float): The standard value for comparison.
    operation (str): The operation to determine the comparison logic, containing one or more of '>', '=', '<'.
    candidate (iterable): A collection of candidate values to be compared with the standard value.

    Returns:
    bool: Returns True if any candidate value satisfies the operation condition with the standard value; otherwise, returns False.
    """
    for target in candidate:
        try:
            result = ">=<"
            if standard > target:
                result = ">"
            elif standard == target:
                result = "="
            else:
                result = "<"
            return result in operation
        except TypeError:
            pass


def get_resources(uid, video, interval, limit):
    """
    * 根据用户ID获取微博资源
    *
    * @param uid 用户ID
    * @param video 是否获取视频资源
    * @param interval 请求间隔时间
    * @param limit 资源筛选的限制条件
    * @return 返回获取到的资源列表
    """
    page = 1
    size = 25
    amount = 0
    total = 0
    empty = 0
    aware = 1
    exceed = False
    resources = []

    while empty < aware and not exceed:
        try:
            url = "https://m.weibo.cn/api/container/getIndex?count={}&page={}&containerid=107603{}".format(
                size, page, uid
            )
            response = request_fit("GET", url, is_api=True)

            if response.status_code == 418:
                print_fit(
                    "anti-scraping triggered (418) at page #{}, backing off...".format(page), pin=True
                )
                empty = aware
                continue

            json_data = json.loads(response.text)
        except Exception as e:
            print_fit("request error at page #{}: {}".format(page, e), pin=True)
            anti_scrape_fail()
        else:
            # 安全访问：反爬或异常响应可能没有 "ok" 或 "data" 字段
            if not isinstance(json_data, dict) or json_data.get("ok") != 1:
                empty = empty + 1
                print_fit("empty or abnormal response at page #{}".format(page), pin=True)
            else:
                empty = 0
                data = json_data.get("data", {})
                if not isinstance(data, dict):
                    empty = empty + 1
                    print_fit("invalid data format at page #{}".format(page), pin=True)
                else:
                    if total == 0 and "cardlistInfo" in data:
                        total = data["cardlistInfo"]["total"]
                    cards = data.get("cards", [])
                    for card in cards:
                        if "mblog" in card:
                            mblog = card["mblog"]
                            if "isTop" in mblog and mblog["isTop"]:
                                continue
                            mid = int(mblog["mid"])
                            date = parse_date(mblog["created_at"])
                            mark = {
                                "mid": mid,
                                "bid": mblog["bid"],
                                "date": date,
                                "text": mblog["text"],
                            }
                            amount += 1
                            if compare(limit[0], ">", [mid, date]):
                                exceed = True
                            if compare(limit[0], ">", [mid, date]) or compare(
                                    limit[1], "<", [mid, date]
                            ):
                                continue
                            if "pics" in mblog:
                                for index, pic in enumerate(mblog["pics"], 1):
                                    if "large" in pic:
                                        resources.append(
                                            merge(
                                                {
                                                    "url": pic["large"]["url"],
                                                    "index": index,
                                                    "type": "photo",
                                                },
                                                mark,
                                            )
                                        )
                            elif "page_info" in mblog and video:
                                if "media_info" in mblog["page_info"]:
                                    media_info = mblog["page_info"]["media_info"]
                                    streams = [
                                        media_info[key]
                                        for key in [
                                            "mp4_720p_mp4",
                                            "mp4_hd_url",
                                            "mp4_sd_url",
                                            "stream_url",
                                        ]
                                        if key in media_info and media_info[key]
                                    ]
                                    if streams:
                                        resources.append(
                                            merge(
                                                {"url": streams.pop(0), "type": "video"}, mark
                                            )
                                        )
            print_fit(
                "{} {}(#{})".format(
                    (
                        "analysing weibos..."
                        if empty < aware and not exceed
                        else "finish analysis"
                    ),
                    progress(amount, total),
                    page,
                ),
                pin=True,
            )
            page += 1
        finally:
            time.sleep(interval)

    print_fit(
        "\npractically scan {} weibos, get {} {}".format(
            amount, len(resources), "resources" if video else "pictures"
        )
    )
    return resources


def format_name(item):
    """
    根据提供的项生成格式化后的名称。

    该函数从项的URL中提取名称部分，并对其进行安全处理，以确保名称中不包含非法字符。
    它还支持通过替换模板中的占位符来动态生成名称。

    参数:
    item (dict): 包含URL和其他信息的字典项。

    返回:
    str: 格式化并经过安全处理的名称。
    """
    item["name"] = re.sub(r"\?\S+$", "", re.sub(r"^\S+/", "", item["url"]))

    def safeify(name):
        """
        将名称中的非法字符转换为安全的替代字符。

        参数:
        name (str): 需要进行安全处理的名称。

        返回:
        str: 经过安全处理的名称。
        """
        template = {
            "\\": "＼",
            "/": "／",
            ":": "：",
            "*": "＊",
            "?": "？",
            '"': "＂",
            "<": "＜",
            ">": "＞",
            "|": "｜",
        }
        for illegal in template:
            name = name.replace(illegal, template[illegal])
        return name

    def substitute(matched):
        """
        替换模板中的占位符。

        参数:
        matched (re.Match): 匹配到的占位符信息。

        返回:
        str: 替换后的字符串。
        """
        key = matched.group(1).split(":")
        if key[0] not in item:
            return ":".join(key)
        elif key[0] == "date":
            return item[key[0]].strftime(key[1]) if len(key) > 1 else str(item[key[0]])
        elif key[0] == "index":
            return str(item[key[0]]).zfill(int(key[1] if len(key) > 1 else "0"))
        elif key[0] == "text":
            return re.sub(r"<.*?>", "", item[key[0]]).strip()
        else:
            return str(item[key[0]])

    return safeify(re.sub(r"{(.*?)}", substitute, args.name))


def download(url, originalpath, overwrite, errorcallback):
    """
    下载文件到指定路径，并根据参数决定是否覆盖现有文件。

    参数:
    url (str): 要下载的文件的URL。
    path (str): 保存下载文件的路径。
    overwrite (bool): 如果文件已存在，是否覆盖文件。

    返回:
    bool: 下载并保存文件成功返回True，否则返回False。
    """
    if os.path.exists(originalpath) and not overwrite:
        return True
    try:
        print_fit("downloading:GET:" + url)
        response = request_fit("GET", url, stream=True)
        if response.status_code != 200:
            print_fit(
                'failed to download "{}" status_code ({})'.format(
                    url, response.status_code
                )
            )
            if errorcallback:
                errorcallback(url)
            return False
        # 先保存到临时文件.part,写入文件，避免下载失败时文件损坏，然后重命名
        pathpart = originalpath + ".downloading.part"
        if not os.path.exists(os.path.dirname(originalpath)):
            os.makedirs(os.path.dirname(originalpath))

        with open(pathpart, "wb") as f:
            for chunk in response.iter_content(chunk_size=512):
                if chunk:
                    f.write(chunk)
        os.rename(pathpart, originalpath)
        print("saved file success:" + originalpath)
    except Exception:
        if os.path.exists(originalpath):
            os.remove(originalpath)
        return False
    else:
        return True


"""这段Python代码主要用于从微博下载图片和视频。具体功能如下：
解析命令行参数：通过argparse库解析用户输入的命令行参数。
获取用户列表：根据命令行参数从用户列表或文件中读取用户信息。
设置保存目录：根据命令行参数设置保存下载内容的目录，如果目录不存在则创建。
解析边界条件：解析用户指定的时间或ID范围，确保范围有效。
初始化线程池：使用concurrent.futures.ThreadPoolExecutor创建线程池，用于并行下载资源。
遍历用户列表：对每个用户执行以下操作：
获取用户的昵称和UID。
获取用户的所有资源（图片和视频）。
创建保存资源的子目录。
使用线程池下载资源，处理下载失败的情况并重试。
输出结果：打印下载状态和结果。"""
args = parser.parse_args(nargs_fit(parser, sys.argv[1:]))
users = []
if args.users:
    users = (
        [user.decode(system_encoding) for user in args.users]
        if is_python2
        else args.users
    )
elif args.files:
    users = [read_from_file(path.strip()) for path in args.files]
    users = reduce(lambda x, y: x + y, users)
users = [user.strip() for user in users]

if args.directory:
    base = args.directory

    if os.path.exists(base):
        if not os.path.isdir(base):
            quit("saving path is not a directory")
    elif args.yes:
        make_dir(base)
    elif confirm('directory "{}" doesn\'t exist, help to create?'.format(base)):
        make_dir(base)
    else:
        quit("do it youself :)")
else:
    base = os.path.join(os.path.dirname(__file__), "weiboPic")
    if not os.path.exists(base):
        make_dir(base)

boundary = args.boundary.split(":")
boundary = boundary * 2 if len(boundary) == 1 else boundary
numberify = lambda x: int(x) if re.search(r"^\d+$", x) else bid_to_mid(x)
dateify = lambda t: datetime.datetime.strptime(t, "@%Y%m%d").date()
parse_point = lambda p: dateify(p) if p.startswith("@") else numberify(p)
try:
    boundary[0] = 0 if boundary[0] == "" else parse_point(boundary[0])
    boundary[1] = float("inf") if boundary[1] == "" else parse_point(boundary[1])
    if type(boundary[0]) == type(boundary[1]):
        assert boundary[0] <= boundary[1]
except:
    quit("invalid id range {}".format(args.boundary))

token = "SUB={}".format(args.cookie) if args.cookie else args.cookieall or None
if token:
    with _session_lock:
        _session.headers["Cookie"] = token
print_fit("initial cookie: {}".format(token or "(none)"))
pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.size)
last_msg = ""
for number, user in enumerate(users, 1):

    print_fit("{}/{} {}".format(number, len(users), time.ctime()))

    if re.search(r"^\d{10}$", user):
        nickname = uid_to_nickname(user)
        uid = user
    else:
        nickname = user
        uid = nickname_to_uid(user)

    if not nickname or not uid:
        print_fit("invalid account {}".format(user))
        print_fit("-" * 30)
        continue

    print_fit("{} {}".format(nickname, uid))

    try:
        resources = get_resources(uid, args.video, args.interval, boundary)
    except KeyboardInterrupt:
        quit()

    album = os.path.join(base, nickname)
    if resources and not os.path.exists(album):
        make_dir(album)

    retry = 0
    while resources and retry <= args.retry:

        if retry > 0:
            print_fit("automatic retry {}".format(retry))

        total = len(resources)
        tasks = []
        done = 0
        failed = {}
        cancel = False

        for resource in resources:
            path = os.path.join(album, format_name(resource))
            tasks.append(
                pool.submit(
                    download, resource["url"], path, args.overwrite,
                    lambda url: print("failed to download {}".format(url))
                )
            )

        while done != total:
            try:
                done = 0
                for index, task in enumerate(tasks):
                    if task.done():
                        done += 1
                        if task.cancelled():
                            continue
                        elif not task.result():
                            failed[index] = ""
                    elif cancel:
                        if not task.cancelled():
                            task.cancel()
                time.sleep(0.5)
            except KeyboardInterrupt:
                cancel = True
                os._exit(1)
            finally:
                if not cancel:
                    new__msg = ("{} {}" + "\n").format(
                        "downloading..." if done != total else "all tasks done",
                        progress(done, total, True),
                    )
                    if new__msg != last_msg:
                        last_msg = new__msg
                        print_fit(last_msg, pin=True)
                else:
                    print_fit(
                        "waiting for cancellation... ({})".format(total - done),
                        pin=True,
                    )

        if cancel:
            quit()
        print_fit(
            "\nsuccess {}, failure {}, total {}".format(
                total - len(failed), len(failed), total
            )
        )

        resources = [resources[index] for index in failed]
        retry += 1

    for resource in resources:
        print_fit("{} failed".format(resource["url"]))
    print_fit("-" * 30)

quit("bye bye")
