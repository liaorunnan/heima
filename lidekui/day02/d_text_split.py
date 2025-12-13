import re

split_sep = ['[。！？......]', '[；]', '[，]']

def text_split(text, max_len=500):
    if len(text) <= max_len:
        return [text]
    paras = []
    for para in re.split(r'\n+', text):
        if len(para) > max_len:
            for i in range(0, len(para), max_len):
                paras.append(para[i:i+max_len])
        else:
            paras.append(para)


if __name__ == '__main__':
    import asyncio
    import aiohttp

    async def task1():
        print('task1')
        await asyncio.sleep(5)
        print('task1 done')

    async def task2():
        print('task2')
        await asyncio.sleep(1)
        print('task2 done')

    async def main():
        await asyncio.gather(task1(), task2())

    asyncio.run(main())
