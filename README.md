# ATRI

> 预告：此仓库将在 2026 年春节前迎来重大更新 😎

一个基于图搜索、向量化语义检索和认知科学的记忆模块。

> [!WARNING]
> 🚧 The repo is under construction, do not use in production 🚧

## Road Map

- [x] 选型
- [x] 向量检索库、图存储库开发
- [x] 总结、事实抽取、入图、搜索等基本功能实现
- [x] WebUI 可视化
- [x] 记忆更新机制
- [ ] 记忆遗忘机制
- [ ] 情绪机制（AstrBotDevs/nova）

## Demo

![0](https://github.com/user-attachments/assets/2e68db43-179f-49dd-bacb-d3bb8ff5c191)


![1](https://github.com/user-attachments/assets/36644f85-2178-495f-8817-ea242e03975d)

## LongMemEval Dataset

参考 https://github.com/xiaowu0162/LongMemEval/

```bash
git clone https://github.com/xiaowu0162/LongMemEval/ --depth 1
cd LongMemEval/data && wget https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s?download=true && mv "longmemeval_s?download=true" longmemeval_s
cd ..
pip install -r requirements-lite.txt
```
