# weiboPicDownloader

## 项目简介

微博图片批量下载工具，支持通过用户昵称获取 UID 后下载其微博图片。

## 技术栈

- **语言**: Python 3
- **主要依赖**: 见 `requirements.txt`

## 项目结构

```
weiboPicDownloader/
├── LICENSE              # 开源许可证
├── README-CN.md         # 中文说明文档
├── README.md            # 英文说明文档
├── requirements.txt     # Python 依赖
├── test_nickname_to_uid.py  # 昵称转UID测试脚本
└── weiboPicDownloader.py    # 主程序入口
```

## 功能特性

- 通过微博用户昵称解析对应的 UID
- 批量下载用户微博中的图片

## 使用方式

1. 安装依赖: `pip install -r requirements.txt`
2. 运行主程序: `python weiboPicDownloader.py`

## 开发规范

- 遵循 PEP 8 代码风格
- 使用 Git 进行版本控制
