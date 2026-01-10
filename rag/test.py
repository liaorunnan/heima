

from rag.reranker import rank

from loguru import logger







def get_num(n):
    result = list(range(0,n,2)) + list(range(n-1,0,-2))
    return result



   
if __name__ == '__main__':

    
    docs = [
        {"parent": "海鸥飞在海滩上"},
        {"parent": "湛蓝的天空点缀着几朵白云，海洋呈现深浅不一的蓝色，波浪轻拍沙滩形成白色浪花，沙滩呈温暖的黄色，细腻柔软。前景中有两个孩子奔跑嬉戏，左侧一位小女孩戴着黄色遮阳帽、穿红色泳衣专注建造沙堡（沙堡上插着小旗子），右侧一对情侣手牵手散步（男士穿蓝T恤绿短裤，女士穿黄上衣橙短裙）。背景有椰子树茂密矗立，树上挂着黄色椰子，沙滩上还有其他游客活动。整体采用卡通风格，线条简洁、色彩明亮饱和，氛围温馨欢乐，充满夏日阳光与度假气息"},
        {"parent": "椰子树在海边高高地长大"},
        {"parent": "这张图片展示了一个充满夏日活力的海滩场景。金黄色的沙滩细腻柔软，上面留有人们活动的脚印；清澈的蓝绿色海水轻柔拍岸，形成白色浪花，营造出宁静诱人的氛围。沙滩上，人们坐在色彩缤纷的沙滩椅上，头顶着红、黄、蓝、绿等多色遮阳伞，身着轻便泳装或夏季服饰，显得格外放松。海水中，一位穿蓝色比基尼的女性正走向岸边，身旁的小孩在水中嬉戏，笑容满面。远处海面上，几艘白色帆船在微风中缓缓航行，为画面增添动感。天空湛蓝澄澈，点缀着悠闲的白云，整体色彩明亮饱和（沙滩金黄、海水蓝绿、遮阳伞多彩），传递出轻松愉悦的度假气息，完美捕捉了夏日海滩的休闲美好瞬间。"},
    ]
    update_query="海滩上有什么"
    docs = sorted(docs, key=lambda x: rank(x['parent'], update_query), reverse=True)
    logger.info(f"搜索结果：{docs}")
    for doc in docs:
        logger.info(f"文档：{doc['parent']}")
        print(rank(doc['parent'], update_query))
    


