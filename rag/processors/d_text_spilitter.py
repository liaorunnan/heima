import re
split_seps = ["[。！？……]","[；]","[，]"]
def split_text(text,max_len=500):
    if len(text) <= max_len:
        return [text]
    parag=[]
    for para in re.split("\n",text):

        if len(para)<=max_len:
            parag.append(para)
            continue
        for sep in split_seps:
            ...
    async def main():
            ...

if __name__ == '__main__':
    # import asyncio
    # import aiohttp
    # async def task1():
    #     print("task1")
    #     await asyncio.sleep(5)
    #     print("task1 done")
    #
    # async def task2():
    #     print("task2")
    #     await asyncio.sleep(1)
    #     print("task2 done")
    #
    # async def main():
    #     await asyncio.gather(task1(),task2(),task1(),task2())
    #
    # asyncio.run(main())
    ...

